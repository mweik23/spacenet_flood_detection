from ml_tools.metrics.core import RunningStats

import pytest
import numpy as np

@pytest.fixture(params=[1, 3, 10, 50])
def window(request) -> int:
    return request.param

@pytest.fixture
def running_stats(window):
    return RunningStats(window=window,
                        phase='train',
                        weight_key='num_pixels',
                        mean_keys=('loss',),
                        sum_keys=("tp","fp","fn","tn"),
                        ddp_sync=False)

@pytest.fixture(scope="module")
def batch_stream():
    rng = np.random.default_rng(12345)
    B = 100  # num batches
    out = []
    for _ in range(B):
        num_pixels = int(rng.integers(10_000, 50_000))
        # sample a confusion vector that sums to num_pixels
        # multinomial keeps invariants perfect
        tp, fp, fn, tn = rng.multinomial(num_pixels, [0.01, 0.01, 0.01, 0.97]).tolist()

        loss = float(rng.normal(loc=0.3, scale=0.05))
        loss = max(0.0, loss)

        out.append(
            dict(
                loss=loss,
                num_pixels=num_pixels,
                tp=tp, fp=fp, fn=fn, tn=tn,
            )
        )
    return out

@pytest.fixture
def expected_rolling_stats(batch_stream, window):
    num_windows = -(-len(batch_stream) // window)  # ceiling division
    averages = []
    for i in range(num_windows):
        start_idx = min(i * window, len(batch_stream)-window)
        end_idx = min((i + 1) * window, len(batch_stream))
        window_batches = batch_stream[start_idx:end_idx]

        total_weight = sum(batch['num_pixels'] for batch in window_batches)
        weighted_loss = sum(batch['loss'] * batch['num_pixels'] for batch in window_batches)
        mean_loss = weighted_loss / total_weight if total_weight > 0 else 0.0

        total_tp = sum(batch['tp'] for batch in window_batches)
        total_fp = sum(batch['fp'] for batch in window_batches)
        total_fn = sum(batch['fn'] for batch in window_batches)
        total_tn = sum(batch['tn'] for batch in window_batches)

        averages.append(dict(
            mean_loss=mean_loss,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            total_tn=total_tn,
        ))

    return averages

@pytest.fixture
def all_time_stats(batch_stream):
    total_weight = sum(batch['num_pixels'] for batch in batch_stream)
    weighted_loss = sum(batch['loss'] * batch['num_pixels'] for batch in batch_stream)
    mean_loss = weighted_loss / total_weight if total_weight > 0 else 0.0

    total_tp = sum(batch['tp'] for batch in batch_stream)
    total_fp = sum(batch['fp'] for batch in batch_stream)
    total_fn = sum(batch['fn'] for batch in batch_stream)
    total_tn = sum(batch['tn'] for batch in batch_stream)

    return dict(
        mean_loss=mean_loss,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        total_tn=total_tn,
    )
    
def test_rolling_stats(running_stats, batch_stream, expected_rolling_stats, window):
    for i, batch in enumerate(batch_stream):
        running_stats.update(batch)

        if (i + 1) % window == 0 or (i + 1) == len(batch_stream):
            expected = expected_rolling_stats[-(-(i + 1) // window) - 1]
            stats = {
                'mean_loss': running_stats.rolling_mean('loss'),
                'total_tp': running_stats.rolling_sum('tp'),
                'total_fp': running_stats.rolling_sum('fp'),
                'total_fn': running_stats.rolling_sum('fn'),
                'total_tn': running_stats.rolling_sum('tn')
            }

            assert np.isclose(stats['mean_loss'], expected['mean_loss']), f"Mean loss mismatch at batch {i+1}"
            assert stats['total_tp'] == expected['total_tp'], f"TP mismatch at batch {i+1}"
            assert stats['total_fp'] == expected['total_fp'], f"FP mismatch at batch {i+1}"
            assert stats['total_fn'] == expected['total_fn'], f"FN mismatch at batch {i+1}"
            assert stats['total_tn'] == expected['total_tn'], f"TN mismatch at batch {i+1}"

def test_all_time_stats(running_stats, batch_stream, all_time_stats):
    for batch in batch_stream:
        running_stats.update(batch)

    stats = {
        'mean_loss': running_stats.epoch_weighted_mean('loss'),
        'total_tp': running_stats.epoch_sum('tp'),
        'total_fp': running_stats.epoch_sum('fp'),
        'total_fn': running_stats.epoch_sum('fn'),
        'total_tn': running_stats.epoch_sum('tn')
    }

    assert np.isclose(stats['mean_loss'], all_time_stats['mean_loss']), "Mean loss mismatch for all-time stats"
    assert stats['total_tp'] == all_time_stats['total_tp'], "TP mismatch for all-time stats"
    assert stats['total_fp'] == all_time_stats['total_fp'], "FP mismatch for all-time stats"
    assert stats['total_fn'] == all_time_stats['total_fn'], "FN mismatch for all-time stats"
    assert stats['total_tn'] == all_time_stats['total_tn'], "TN mismatch for all-time stats"

    