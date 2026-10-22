"""Public naming and preflight must preserve the existing scientific presets."""

from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from tr_asbs.check import configured_data, inspect_configuration, main
from tr_asbs.train import build_dispatch


CONFIG = Path(__file__).resolve().parents[1] / "src/tr_asbs/config"
PRESETS = ["gmm_mode", "gmm_dim", "mw5", "mw5_delta", "lj13", "lj38", "lj55",
           "morse13_asbs", "morse13_ce_fast", "ad_ic_gamma_asbs", "ad_ic_reference",
           "ala2_gamma_asbs_iw", "ala2_mlp_gamma_asbs_iw"]


@pytest.mark.parametrize("experiment", PRESETS)
@pytest.mark.parametrize("method", ["asbs-iw", "tr-sbs"])
def test_public_method_does_not_replace_target_model_or_solver(experiment, method):
    dispatch = build_dispatch(method, experiment)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG)):
        historical = compose(config_name="train", overrides=[f"experiment={experiment}"])
        public = compose(config_name="train", overrides=list(dispatch.overrides))
    for field in ["potential", "controller", "corrector", "source", "base_sde",
                  "integrator", "terminal_cost"]:
        assert OmegaConf.to_container(public[field]) == OmegaConf.to_container(historical[field])
    assert public.paper_method == dispatch.method.name
    assert public.gamma == historical.gamma
    assert public.gamma_schedule == dispatch.method.gamma_schedule


def test_missing_reference_is_reported_without_instantiating_energy(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--experiment", "lj13", "--data-root", str(tmp_path)])
    assert error.value.code == 1
    output = capsys.readouterr().out
    assert '"exists": false' in output
    assert '"training_performed": false' in output


def test_alanine_data_fields_include_calibration_and_validation():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG)):
        config = compose(config_name="train", overrides=["experiment=ad_ic_gamma_asbs"])
    inputs = dict(configured_data(config))
    assert inputs["potential.transform_data_path"].endswith("alanine_dipeptide/val.h5")
    assert inputs["potential.ref_samples_path"].endswith("alanine_dipeptide/val.h5")


def test_no_ground_truth_target_does_not_require_substitute_reference(tmp_path):
    result = inspect_configuration(build_dispatch("tr-sbs", "morse13_asbs"), data_root=tmp_path)
    assert result["data"] == []
    assert result["support_error"] is None
    unsupported = inspect_configuration(build_dispatch("pis", "lj38"), data_root=tmp_path)
    assert unsupported["support_error"]


@pytest.mark.parametrize("config_name", ["train", "variational_train"])
def test_portable_logging_and_output_defaults(config_name, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("TR_ASBS_ROOT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG)):
        config = compose(config_name=config_name, overrides=["experiment=gmm_mode"])
    assert config.root == "."
    assert config.wandb_mode == "offline"
    assert config.fabric.loggers.entity is None


@pytest.mark.parametrize("method", ["asbs", "gamma-asbs", "tr-asbs", "asbs-iw", "tr-sbs", "pis", "dis"])
def test_real_public_runtime_can_print_hydra_config_without_training(method, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "tr_asbs.train", "--method", method,
         "--experiment", "gmm_mode", "--cfg", "job", "--resolve"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    config = OmegaConf.create(result.stdout)
    assert config.method_id == method
    assert not list(tmp_path.iterdir())
