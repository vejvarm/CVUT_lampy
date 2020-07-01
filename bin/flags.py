class FLAGS:
    naccs = 6
    nlamps = 1  # for new X_li.npy files use 1, for old X.npy use 3
    naccs_per_lamp = 2

    paths = {
        "training": {"folder": "data/trening",
                     "dataset": ["neporuseno", "neporuseno2", "poruseno"],
                     "period": "2months"
                     },
        "validation": {"folder": "data/validace",
                       "dataset": ["neporuseno", "poruseno"],
                       "period": ""
                       },
        "serialized": {"folder": "data/serialized",
                       "dataset": [""],
                       "period": ""
                       }
    }

    serialized = {"unbroken": 86,
                  "broken": 140}

    lamps = {
        "l1": ("X_l1.npy", "y_l1.npy"),
        "l2": ("X_l2.npy", "y_l2.npy"),
        "l3": ("X_l3.npy", "y_l3.npy"),
    }

    preprocessing = {
        "use_autocorr": True,
        "rem_neg": True
    }

    PSNR_csv_setup = {
        "name": "PSNR.csv",
        "name2": "PSNRda.csv",
        "columns": ["episode", "signal_amp_max", "noise_amp_max", "train_PSNR (dB)", "test_PSNR (dB)"],
        "sep": ";",
        "decimal": ",",
        "index": False,
        "line_terminator": "\n"
    }
