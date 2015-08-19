import numpy
from fisherman import data_io
from itertools import imap


class SequenceAmplifier(object):
    """
    Applys a sequence of functions to a sequence of images, returning a sequence consisting
     of each input item, and the result of each function on each input item.
    """
    
    def __init__(self, amplifier_fcns=[]):
        self.amplifier_fcns = []
        self.add_amplifier_fcn(amplifier_fcns)

    def add_amplifier_fcn(self, amplifiers):
        """
        Adds an amplifier function, or a sequence of amplifier functions, to the currently
         existing sequence of amplifier functions
        """
        if callable(amplifiers):
            self.amplifier_fcns.append(amplifiers)
        else:
            self.amplifier_fcns += list(amplifiers)

    def amplify(self, seq):
        """
        Applies the sequence of amplifier functions to a sequence of items
        """
        for item in seq:
            yield item

            for amp in self.amplifier_fcns:
                yield amp(item)


class TrainingExampleAmplifier(SequenceAmplifier):
    """
    An SequenceAmplifier that applies amplifiers to images in a seqence of TrainingExamples,
     returning a sequence of TrainingExamples with the same labels
    """
    
    def amplify(self, seq):
        """
        Applies the sequence of amplifier functions to a sequence of TrainingExamples
        """
        for training_example in seq:
            amp_no = 0
            new_tag = str(training_example.tag) + '_amp%d' 

            training_example.tag = new_tag % amp_no
            yield training_example

            for amp in self.amplifier_fcns:
                amp_no += 1
                yield data_io.TrainingExample(
                    amp(training_example.image),
                    training_example.label,
                    tag=new_tag % amp_no
                )
            

class AmplifierChain(object):

    def __init__(self, amplifiers=[]):
        self.amplifiers = list(amplifiers)

    def add_amplifier(self, amp):
        self.amplifiers.append(amp)
    
    def amplify(self, seq):
        for amplifier in self.amplifiers:
            seq = amplifier.amplify(seq)

        for item in seq:
            yield item
