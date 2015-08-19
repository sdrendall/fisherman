from nose import *
from fisherman import amplification as amp

def test_sequence_amplifier():
    seq = (1, 2, 3, 4, 5)
    amp_fcns = [double, triple]
    amplifier = amp.SequenceAmplifier(amp_fcns)

    res_seq = amplifier.amplify(seq)

    exp_seq = []
    for item in seq:
        exp_seq.append(item)
        for fcn in amp_fcns:
            exp_seq.append(fcn(item))

    for (res, exp) in zip(res_seq, exp_seq):
        assert res == exp


def test_chain_amplifier():
    seq = (1, 2, 3, 4, 5)
    amp1 = amp.SequenceAmplifier()
    amp2 = amp.SequenceAmplifier()

    amp1.add_amplifier_fcn(double)
    amp2.add_amplifier_fcn(triple)

    chain_amp = amp.AmplifierChain([amp1, amp2])

    exp_seq = amp2.amplify(amp1.amplify(seq))
    res_seq = chain_amp.amplify(seq)

    for (res, exp) in zip (res_seq, exp_seq):
        assert res == exp


def double(x):
    return x * 2

def triple(x):
    return x * 3
