from ecd_platform import ECDPipeline, ECDConfig, ImagFreqPolicy

config = ECDConfig(
    opt_dir='opt_conf',
    ecd_dir='ecd_opt_60_roots',
    exp_file='87.csv',
    sigma=0.1,
    shift=0.15,
    scale_factor=5.0,
    imag_freq_policy=ImagFreqPolicy.TOLERANT,
    imag_freq_threshold=-10.0,

    # ── 波长范围 ──
    wavelength_range=(200, 400),      # 只显示 200-400 nm

    # ── 实验谱平滑 ──
    smooth_method='fft',              # 平滑方法: 'fft' / 'savgol' / 'none'
    smooth_factor=0.1,                # FFT 截止比例，越小越平滑
)

pipeline = ECDPipeline(config)
pipeline.run()