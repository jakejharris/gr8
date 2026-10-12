use g6kvman::pool::{Namespace, STOCK_NAMESPACE};
use g6kvman::{Finished, Pool, Rules, Why};

fn namespace(tag: u8) -> Namespace {
    let mut namespace = STOCK_NAMESPACE;
    namespace[0] = tag;
    namespace
}

#[test]
fn retained_prefixes_only_match_their_exact_steering_namespace() {
    let mut pool = Pool::new(16, 16, 16 * 64).unwrap();
    pool.keep(Rules { strict: false, noend: true, grid: 0 });

    let prompt: Vec<i32> = (0..96).collect();
    let run = pool.reserve(96, 128).unwrap();
    pool.retain_namespaced(
        &run,
        Finished { history: prompt.clone(), valid: 95, written_end: 96, pf_end: 96 },
        namespace(1),
    )
    .unwrap();

    let exact = pool.match_prefix_namespaced(&prompt, 96, 128, 95, namespace(1));
    assert_eq!(exact.why, Why::Hit);
    assert!(exact.skip > 0);

    for wrong in [STOCK_NAMESPACE, namespace(2)] {
        let miss = pool.match_prefix_namespaced(&prompt, 96, 128, 95, wrong);
        assert_eq!(miss.skip, 0);
        assert_eq!(miss.why, Why::Namespace);
    }
}
