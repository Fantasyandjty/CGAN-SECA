class hparams:
    def __init__(self):

        self.train_scp = "scp/train_segan.scp"
        self.valid_ratio = 0.05
        self.fs = 16000
        self.win_len = 16384
        self.emph_coeff = 0.95


        self.n_epoch = 100


        self.batch_size = 64
        self.ref_batch_size = 128
        self.num_workers = 2


        self.lr_G = 2e-4
        self.lr_D = 2e-4
        self.optimizer = "rmsprop"
        self.grad_clip = 5.0

        self.size_z = (1024, 8)


        self.loss_mode = "improved"
        self.w_adv = 0.5
        self.w_l1 = 100.0


        self.w_stft = 0.0
        self.stft_fft_sizes = (512, 1024, 2048)
        self.stft_hop_sizes = (128, 256, 512)
        self.stft_win_lengths = (512, 1024, 2048)


        self.use_seca = True


        self.se_init = "identity"
        self.use_vbn = True
        self.use_dropout = True
        self.dropout_p = 0.5


        self.seed = 1234
        self.deterministic = True


        self.save_path = "save"
        self.path_save = "save"
        self.log_path = "logs"


        self.test_clean_dir = "voicedemand_16k/test/clean"
        self.test_noisy_dir = "voicedemand_16k/test/noisy"
        self.eval_overlap = 0.5
        self.eval_snr_filter = None

        self._apply_env_overrides()

    def _apply_env_overrides(self):
        import os
        prefix = "CGANSECA_"
        for k, v in os.environ.items():
            if not k.startswith(prefix):
                continue
            name = k[len(prefix):]
            if not hasattr(self, name):
                continue
            cur = getattr(self, name)
            try:
                if isinstance(cur, bool):
                    setattr(self, name, v.lower() in ("1", "true", "yes"))
                elif isinstance(cur, int):
                    setattr(self, name, int(v))
                elif isinstance(cur, float):
                    setattr(self, name, float(v))
                else:
                    setattr(self, name, v)
            except ValueError:
                raise ValueError("环境变量 %s 的值 %r 无法转换为 %s" % (k, v, type(cur).__name__))

    def tag(self):
        parts = []
        parts.append("SECA" if self.use_seca else "noSECA")
        parts.append("VBN" if self.use_vbn else "noVBN")
        parts.append("DO" if self.use_dropout else "noDO")
        parts.append(self.loss_mode)
        if getattr(self, "se_init", "identity") != "identity":
            parts.append("seinit-" + self.se_init)
        parts.append("s%d" % self.seed)
        return "_".join(parts)
