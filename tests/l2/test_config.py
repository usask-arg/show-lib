from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk
import yaml

from showlib.l2.config import altitude_grid, engine_kwargs, load_retrieval_config

REQUIRED = {"preprocessor": {"wavenumber_cminv": 7335.165}}


def test_preprocessor_wavenumber_is_required():
    with pytest.raises(ValueError, match=r"preprocessor\.wavenumber_cminv"):
        load_retrieval_config()


def test_defaults_are_the_previous_settings():
    config = load_retrieval_config(REQUIRED)
    assert config["minimizer"]["max_nfev"] == 20
    assert config["observation"]["upper_bound_m"] == 30000
    h2o = "absorbers", "h2o", "tikh_factor"
    preprocessor_state = config["preprocessor"]["state"]
    assert preprocessor_state[h2o[0]][h2o[1]][h2o[2]] == 2.5e3
    assert config["retrieval"]["state"][h2o[0]][h2o[1]][h2o[2]] == 2.5e1


def test_noise_correlation_is_on_by_default():
    assert load_retrieval_config(REQUIRED)["retrieval"]["noise_correlation"] is True


def test_overrides_are_merged_into_the_defaults():
    config = load_retrieval_config({**REQUIRED, "minimizer": {"max_nfev": 50}})
    assert config["minimizer"]["max_nfev"] == 50
    assert config["minimizer"]["method"] == "trf"
    assert config["preprocessor"]["wavenumber_cminv"] == 7335.165


def test_unknown_settings_are_rejected():
    with pytest.raises(ValueError, match=r"minimizer\.max_nfe"):
        load_retrieval_config({**REQUIRED, "minimizer": {"max_nfe": 50}})


def test_state_sections_accept_new_keys():
    config = load_retrieval_config(
        {
            **REQUIRED,
            "retrieval": {"state": {"absorbers": {"h2o": {"initial_guess": 1e-6}}}},
        }
    )
    h2o = config["retrieval"]["state"]["absorbers"]["h2o"]
    assert h2o["initial_guess"] == 1e-6
    assert h2o["tikh_factor"] == 2.5e1


def test_loading_does_not_modify_the_defaults():
    load_retrieval_config({**REQUIRED, "minimizer": {"max_nfev": 50}})
    assert load_retrieval_config(REQUIRED)["minimizer"]["max_nfev"] == 20


def test_loads_a_yaml_file(tmp_path):
    path = tmp_path / "retrieval.yaml"
    path.write_text(yaml.safe_dump({**REQUIRED, "observation": {"row_reduction": 2}}))
    assert load_retrieval_config(path)["observation"]["row_reduction"] == 2


def test_altitude_grid_matches_the_previous_default():
    expected = np.unique(
        np.concatenate(
            (
                np.arange(0, 9000.0, 1000.0),
                np.arange(9000, 35000, 500.0),
                np.arange(35000, 40000, 1000),
                np.arange(40000, 65001, 5000),
            )
        )
    )
    np.testing.assert_array_equal(
        altitude_grid(load_retrieval_config(REQUIRED)), expected
    )


def test_engine_kwargs_combine_shared_and_stage_settings():
    config = load_retrieval_config({**REQUIRED, "engine": {"num_threads": 8}})

    preprocessor = engine_kwargs(config, "preprocessor")
    retrieval = engine_kwargs(config, "retrieval")

    assert preprocessor["num_threads"] == 8
    assert preprocessor["multiple_scatter_source"] is sk.MultipleScatterSource.TwoStream
    assert "wavelength_batch_size" not in preprocessor
    assert retrieval["wavelength_batch_size"] == 128
