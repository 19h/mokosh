#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[test]
fn required_avx2_path_is_executed() {
    use mokosh::utils::simd;
    if std::env::var_os("MOKOSH_REQUIRE_AVX2").is_some() {
        assert!(
            simd::has_avx2(),
            "This validation run requires AVX2 execution"
        );
        assert!(
            simd::has_fma(),
            "This validation run requires FMA execution"
        );
    }
    let input: Vec<f32> = (0..257).map(|i| (i as f32 - 128.0) / 16.0).collect();
    let mut out = vec![0.0; input.len()];
    simd::exp_batch(&input, &mut out);
    for (a, b) in input.iter().zip(&out) {
        assert!((b - a.exp()).abs() / a.exp() < 0.04);
    }
    let mut result = vec![0.25; input.len()];
    simd::fmadd_scalar_batch(&input, 0.3, &mut result);
    for (&a, &b) in input.iter().zip(&result) {
        assert!((b - (a * 0.3 + 0.25)).abs() < 1e-6);
    }
}
