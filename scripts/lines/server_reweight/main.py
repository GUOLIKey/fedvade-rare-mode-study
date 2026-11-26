"""Minimal, auditable patches to the author's model.py.

sample_floor > 0 : gmm_sampling allocates at least `sample_floor` points to every
                   pooled local component before the remainder goes by global pi.
usage_floor  > 0 : ELBO_Loss_server adds  beta * sum_k relu(floor - mean_i yita_ik)^2
                   so no server component is starved of responsibility.
"""
from pathlib import Path


def _once(source, old, new):
    assert source.count(old) == 1, "expected one match for: %r" % old
    return source.replace(old, new)


def apply(path: Path, sample_floor: int, usage_floor: float, beta: float) -> None:
    src = path.read_text()

    if sample_floor > 0:
        old = "    y_hat = np.random.choice(K, size=nums, p=global_pi)"
        new = """    _floor = %d
    if _floor * K > nums:
        raise ValueError('sample floor exceeds total sampling budget')
    if _floor * K <= nums:
        _counts = np.full(K, _floor)
        _counts += np.random.multinomial(nums - _floor * K, global_pi)
    y_hat = np.repeat(np.arange(K), _counts)
    np.random.shuffle(y_hat)""" % sample_floor
        src = _once(src, old, new)

    if usage_floor > 0.0:
        head, tail = src.split("def ELBO_Loss_server", 1)
        old = "        return Loss"
        new = """        _usage = yita_c.mean(0)
        Loss = Loss + %r * torch.sum(torch.relu(%r - _usage) ** 2)
        return Loss""" % (beta, usage_floor)
        assert tail.count(old) == 1
        tail = tail.replace(old, new, 1)
        src = head + "def ELBO_Loss_server" + tail

    path.write_text(src)
