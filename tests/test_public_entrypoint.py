"""Publication launcher contracts; no training or optional runtime imports."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tr_asbs.methods import METHODS, get_method
from tr_asbs.train import build_dispatch, main


@pytest.mark.parametrize(
    ("key", "trainer", "schedule", "algorithm"),
    [
        ("asbs", "train_asbs", None, None),
        ("gamma-asbs", "train_gamma_asbs", "fixed", None),
        ("tr-asbs", "train_gamma_asbs", "trust_region", None),
        ("asbs-iw", "train_gamma_asbs_iw", "fixed", None),
        ("tr-sbs", "train_gamma_asbs_iw", "trust_region", None),
        ("pis", "train_variational", None, "pis"),
        ("dis", "train_variational", None, "dis"),
    ],
)
def test_paper_names_select_compatible_runtime(key, trainer, schedule, algorithm):
    dispatch = build_dispatch(key, "gmm_mode", ["seed=2", "gamma=0.4"])
    assert dispatch.module == f"tr_asbs.experiment.{trainer}"
    assert dispatch.method.gamma_schedule == schedule
    assert dispatch.method.algorithm == algorithm
    assert "seed=2" in dispatch.overrides
    assert "gamma=0.4" in dispatch.overrides
    assert dispatch.overrides[0] == "experiment=gmm_mode"
    if schedule is not None:
        assert f"gamma_schedule={schedule}" in dispatch.overrides
    if algorithm is not None:
        assert f"algorithm={algorithm}" in dispatch.overrides


def test_gamma_is_inherited_until_the_user_overrides_it():
    dispatch = build_dispatch("Gamma-ASBS", "mw5")
    assert dispatch.method.name == "Gamma-ASBS"
    assert "gamma_schedule=fixed" in dispatch.overrides
    assert not any(value.startswith("gamma=") for value in dispatch.overrides)
    assert get_method("TR_ASBS").variant_of == "Gamma-ASBS"


@pytest.mark.parametrize("method", list(METHODS))
def test_canonical_names_are_saved_in_the_actual_hydra_config(method):
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[1] / "src" / "tr_asbs" / "config"
    dispatch = build_dispatch(method, "gmm_mode", ["name=custom-run"])
    config_name = "variational_train" if method in {"pis", "dis"} else "train"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(dispatch.overrides))
    assert cfg.paper_method == METHODS[method].name
    assert cfg.method_id == method
    assert cfg.method_variant_of == METHODS[method].variant_of
    assert cfg.name == "custom-run"
    if method in {"pis", "dis"}:
        assert cfg.algorithm.name == method
        assert cfg.algorithm.method._target_ == f"tr_asbs.variational.{method.upper()}Method"


@pytest.mark.parametrize("method", ["asbs-iw", "tr-sbs"])
def test_atomic_alanine_uses_unchanged_atomic_trainer(method):
    dispatch = build_dispatch(method, "ala2_gamma_asbs_iw")
    assert dispatch.backend == "atomic"
    assert dispatch.module.endswith("train_gamma_asbs_iw_ala2")


def test_flat_alanine_keeps_its_existing_runtime():
    dispatch = build_dispatch("tr-sbs", "ad_ic_gamma_asbs")
    assert dispatch.backend == "flat"
    assert dispatch.module.endswith("train_gamma_asbs_iw")


@pytest.mark.parametrize("experiment", ["ala2", "ala4"])
@pytest.mark.parametrize("method", ["asbs-iw", "tr-sbs"])
def test_legacy_atomic_curricula_cannot_be_advertised_as_public_runs(experiment, method):
    with pytest.raises(ValueError, match="Use --experiment ala2_gamma_asbs_iw"):
        build_dispatch(method, experiment)


@pytest.mark.parametrize(
    ("method", "experiment", "backend"),
    [
        ("gamma-asbs", "ala2_gamma_asbs_iw", "auto"),
        ("pis", "ala2", "auto"),
        ("tr-sbs", "ala2", "flat"),
        ("tr-sbs", "gmm_mode", "atomic"),
    ],
)
def test_incompatible_atomic_routes_fail_before_training(method, experiment, backend):
    with pytest.raises(ValueError):
        build_dispatch(method, experiment, backend=backend)


@pytest.mark.parametrize(
    ("method", "override"),
    [
        ("tr-sbs", "gamma_schedule=fixed"),
        ("asbs-iw", "++gamma_schedule=trust_region"),
        ("gamma-asbs", "~gamma_schedule"),
        ("tr-asbs", "~gamma_schedule=trust_region"),
        ("pis", "algorithm=dis"),
        ("dis", "algorithm=pis"),
        ("tr-sbs", "experiment=mw5"),
        ("tr-sbs", "paper_method=ASBS-IW"),
        ("tr-sbs", "+method_id=asbs"),
        ("tr-sbs", "++method_id=asbs-iw"),
        ("tr-sbs", "~paper_method"),
        ("tr-asbs", "method_variant_of=ASBS-IW"),
        ("tr-sbs", "~method_variant_of=null"),
        ("pis", "algorithm.name=dis"),
        ("dis", "+algorithm.name=pis"),
        ("pis", "algorithm.method._target_=tr_asbs.variational.DISMethod"),
        ("dis", "++algorithm.method._target_=tr_asbs.variational.PISMethod"),
        ("pis", "~algorithm.method._target_"),
        ("pis", "algorithm.method={_target_:tr_asbs.variational.DISMethod}"),
        ("dis", "~algorithm.method"),
    ],
)
def test_canonical_method_cannot_be_changed_by_hydra_override(method, override):
    with pytest.raises(ValueError):
        build_dispatch(method, "gmm_mode", [override])


def test_matching_identity_override_is_not_duplicated():
    dispatch = build_dispatch("tr-sbs", "gmm_mode", ["gamma_schedule=trust_region"])
    assert dispatch.overrides.count("gamma_schedule=trust_region") == 1


def test_variational_non_identity_overrides_are_preserved():
    overrides = ["algorithm.num_steps=17", "algorithm.model.score_hint_mode=target"]
    dispatch = build_dispatch("pis", "gmm_mode", overrides)
    assert all(override in dispatch.overrides for override in overrides)


@pytest.mark.parametrize(
    "override",
    [
        "experiment@_global_=mw5",
        "+experiment@_global_=mw5",
        "~experiment@_global_",
        "algorithm@_global_=dis",
        "++algorithm@algorithm=dis",
        "+method_id@_global_=asbs",
        "defaults=[_self_,{experiment:mw5}]",
    ],
)
def test_packaged_defaults_cannot_replace_public_method_controls(override):
    with pytest.raises(ValueError, match="Cannot replace public method defaults"):
        build_dispatch("pis", "gmm_mode", [override])


def test_unknown_config_and_method_have_actionable_errors():
    with pytest.raises(ValueError, match="Unknown experiment"):
        build_dispatch("tr-sbs", "does-not-exist")
    with pytest.raises(ValueError, match="Unknown method"):
        build_dispatch("does-not-exist", "gmm_mode")


def test_dry_run_reports_dispatch_without_importing_a_trainer(capsys):
    with patch("tr_asbs.train.importlib.import_module") as importer:
        main(["--method", "tr-asbs", "--experiment", "gmm_mode", "--dry-run", "seed=3"])
    importer.assert_not_called()
    result = json.loads(capsys.readouterr().out)
    assert result["method"] == "TR-ASBS"
    assert result["variant_of"] == "Gamma-ASBS"
    assert "seed=3" in result["overrides"]


def test_main_passes_hydra_arguments_to_the_original_entrypoint():
    original_argv = sys.argv
    captured = []
    fake_runtime = SimpleNamespace(main=lambda: captured.extend(sys.argv))
    with patch("tr_asbs.train.importlib.import_module", return_value=fake_runtime) as importer:
        main(["--method", "tr-sbs", "--experiment", "mw5", "--cfg", "job"])
    importer.assert_called_once_with("tr_asbs.experiment.train_gamma_asbs_iw")
    assert captured == [
        "tr_asbs.experiment.train_gamma_asbs_iw",
        "experiment=mw5",
        "++paper_method=TR-SBS",
        "++method_id=tr-sbs",
        "++method_variant_of=null",
        "gamma_schedule=trust_region",
        "--cfg",
        "job",
    ]
    assert sys.argv is original_argv


def test_listing_and_dry_run_work_without_site_packages():
    # -S excludes site-packages, proving that the public inspection API does
    # not accidentally import Torch, Hydra, Lightning or molecular packages.
    source_dir = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(source_dir)!r}); "
        "from tr_asbs.train import main; "
        "main(['--list-methods']); "
        "main(['--method','tr-sbs','--experiment','gmm_mode','--dry-run']); "
        "assert not any(name in sys.modules for name in ['torch','hydra','lightning','openmm'])"
    )
    result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Gamma-ASBS" in result.stdout
    assert '"method": "TR-SBS"' in result.stdout
    assert len(METHODS) == 7
