import lxmls.readers.simple_sequence as ssr

simple = ssr.SimpleSequence()

# print simple.train
# print simple.test

print "sequence"
for sequence in simple.train.seq_list:
    print sequence

print "x"
for sequence in simple.train.seq_list:
    print sequence.x

print "y"
for sequence in simple.train.seq_list:
    print sequence.y

print "nr"
for sequence in simple.train.seq_list:
    print sequence.nr
