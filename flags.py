class FLAGS:
    naccs = 6
    nlamps = 3
    naccs_per_lamp = 2

    paths = {
        "training": {"folder": "data/trening",
                     "dataset": ["neporuseno", "neporuseno2", "poruseno"],
                     "period": "2months"
                     },
        "validation": {"folder": "data/validace",
                       "dataset": ["neporuseno", "poruseno"],
                       "period": ""
                       }
    }

    preprocessing = {
        "use_autocorr": True,
        "rem_neg": True
    }
