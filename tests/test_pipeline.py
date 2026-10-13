import numpy as np
from PIL import Image

from ca_pipeline import manifest, qa
from ca_pipeline.degrade import bc1_roundtrip, bc3_alpha_roundtrip, degrade, random_crops
from ca_pipeline.textures import (
    LanczosUpscaler, fit_to_engine, looks_like_normal_map, process_texture,
    renormalize_normal_map, wrap_pad,
)


def _noise(w, h, mode="RGB", seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, len(mode)), dtype=np.uint8), mode)


def test_guess_category_and_role():
    assert manifest.guess_category("Tex/Weapons/AK47/ak47.dtx") == "weapon"
    assert manifest.guess_category("Interface/HUD/ammo.dtx") == "ui"
    assert manifest.guess_category("Worlds/Lightmaps/lm_01.dtx") == "lightmap"
    assert manifest.guess_role("ak47_n.png") == "normal"
    assert manifest.guess_role("ak47.png") == "diffuse"


def test_scan_reads_images_and_sorts_by_priority(tmp_path):
    (tmp_path / "Worlds").mkdir()
    (tmp_path / "Weapons").mkdir()
    _noise(64, 32, "RGBA").save(tmp_path / "Worlds" / "wall.png")
    _noise(16, 16).save(tmp_path / "Weapons" / "m4.png")
    (tmp_path / "Weapons" / "m4.dtx").write_bytes(b"\0" * 10)
    recs = manifest.scan(tmp_path)
    assert [r.path for r in recs] == ["Weapons/m4.dtx", "Weapons/m4.png", "Worlds/wall.png"]
    wall = recs[-1]
    assert (wall.width, wall.height, wall.has_alpha, wall.category) == (64, 32, True, "world")
    assert "convert" in recs[0].note
    manifest.write_csv(recs, tmp_path / "m.csv")
    assert (tmp_path / "m.csv").read_text().count("\n") == 4


def test_process_texture_keeps_alpha_pow2_and_passes_qa():
    src = _noise(48, 24, "RGBA")
    out = process_texture(src, LanczosUpscaler(4), tiling=True, pad=4)
    assert out.mode == "RGBA" and out.size == (256, 128)
    assert qa.check_texture(src, out) == []


def test_alpha_cutout_is_binary():
    out = process_texture(_noise(16, 16, "RGBA"), LanczosUpscaler(2), alpha_cutout=True)
    assert set(np.unique(np.asarray(out)[..., 3])) <= {0, 255}


def test_fit_to_engine_clamps_and_snaps():
    assert fit_to_engine(Image.new("RGB", (4096, 1000))).size == (2048, 512)
    assert fit_to_engine(Image.new("RGB", (300, 120))).size == (256, 128)


def test_normal_map_handling():
    rgb = np.random.default_rng(1).random((8, 8, 3), dtype=np.float32)
    v = renormalize_normal_map(rgb) * 2 - 1
    assert np.allclose(np.linalg.norm(v, axis=-1), 1, atol=1e-5)
    flat = np.zeros((8, 8, 3), np.float32) + [0.5, 0.5, 1.0]
    assert looks_like_normal_map(flat) and not looks_like_normal_map(rgb)
    out = process_texture(Image.fromarray((flat * 255).astype(np.uint8)), LanczosUpscaler(2), role="normal")
    assert qa.check_texture(Image.new("RGB", (8, 8)), out, role="normal") == []


def test_wrap_pad_uses_opposite_edge():
    a = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    p = wrap_pad(a, 1)
    assert p[1, 0, 0] == a[0, -1, 0] and p.shape == (6, 6, 1)


def test_qa_flags_problems():
    issues = {i.check for i in qa.check_texture(Image.new("RGBA", (64, 64)), Image.new("RGB", (100, 50)))}
    assert issues == {"power_of_two", "aspect_ratio", "alpha_lost"}
    assert qa.texture_bytes(1024, 1024, "DXT1") == 699050
    assert qa.pick_format(True) == "DXT5" and qa.pick_format(True, alpha_cutout=True) == "DXT1"


def test_obj_stats_and_budget(tmp_path):
    p = tmp_path / "m.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\nf 1 2 3\n")
    s = qa.obj_stats(p)
    assert (s.vertices, s.faces, s.triangles) == (4, 2, 3)
    assert qa.check_mesh(s, qa.MeshStats(0, 0, 7))[0].check == "triangle_budget"
    assert qa.check_mesh(s, qa.MeshStats(0, 0, 6)) == []


def test_bc1_roundtrip_limits_each_block_to_four_colours():
    rgb = np.random.default_rng(2).random((10, 6, 3), dtype=np.float32)
    out = bc1_roundtrip(rgb)
    assert out.shape == rgb.shape
    block = out[:4, :4].reshape(-1, 3)
    assert len(np.unique(block, axis=0)) <= 4
    assert np.abs(out - rgb).mean() < 0.3
    flat = np.full((4, 4, 3), 0.5, np.float32)
    assert np.allclose(bc1_roundtrip(flat), flat, atol=1 / 31)


def test_bc3_alpha_and_degrade():
    a = np.random.default_rng(3).random((8, 8, 1), dtype=np.float32)
    assert np.abs(bc3_alpha_roundtrip(a) - a).max() <= 1 / 14 + 1e-3
    lr = degrade(_noise(64, 64, "RGBA"), scale=4, rng=np.random.default_rng(0))
    assert lr.size == (16, 16) and lr.mode == "RGBA"
    crops = random_crops(_noise(100, 80), 32, 5, np.random.default_rng(0))
    assert len(crops) == 5 and all(c.size == (32, 32) for c in crops)
    assert random_crops(_noise(10, 10), 32, 5) == []


def test_target_scale_downsamples_4x_model_output():
    out = process_texture(_noise(32, 32), LanczosUpscaler(4), target_scale=2)
    assert out.size == (64, 64)


def test_checkpoint_sync_roundtrip(tmp_path):
    import os
    from ca_pipeline.sync import CheckpointSync, LocalRemote, newest_checkpoints, session_deadline

    remote = LocalRemote(tmp_path / "remote")
    run = tmp_path / "run1"
    (run / "training_states").mkdir(parents=True)
    for i in range(3):
        f = run / "training_states" / f"{i}.state"
        f.write_text(str(i))
        os.utime(f, (i, i))
    assert [p.name for p in newest_checkpoints(run, keep=2)] == ["2.state", "1.state"]

    sync = CheckpointSync(remote, run, prefix="runs/x", interval=3600)
    sync.start()
    (run / "models").mkdir()
    (run / "models" / "net_g_5000.safetensors").write_text("w")
    pushed = sync.stop()
    assert {p.name for p in pushed} == {"2.state", "1.state", "net_g_5000.safetensors"}
    assert sync.push_new() == []  # nothing new, nothing re-uploaded

    fresh = CheckpointSync(remote, tmp_path / "run2", prefix="runs/x")
    fresh.pull()  # a new session resumes from what the old one pushed
    assert (tmp_path / "run2" / "models" / "net_g_5000.safetensors").read_text() == "w"
    assert fresh.push_new() == []
    assert session_deadline(0, 12, 45) == 12 * 3600 - 45 * 60


def test_blackwell_issues():
    from ca_pipeline.gpu import blackwell_issues

    assert blackwell_issues((12, 0), "12.8", ["sm_90", "sm_120"]) == []
    assert len(blackwell_issues((12, 0), "12.4", ["sm_90"])) == 2
    assert blackwell_issues((12, 0), None, []) != []
    assert blackwell_issues((8, 0), "11.8", ["sm_80"]) == []  # Ampere: not our concern


def test_dataset_folder_upload_and_pull(tmp_path):
    from ca_pipeline.sync import LocalRemote

    ds = tmp_path / "dataset" / "train" / "hr"
    ds.mkdir(parents=True)
    (ds / "a.png").write_text("x")
    remote = LocalRemote(tmp_path / "remote")
    remote.upload_folder(tmp_path / "dataset", "dataset")
    remote.download("dataset", tmp_path / "pulled")
    assert (tmp_path / "pulled" / "train" / "hr" / "a.png").exists()
