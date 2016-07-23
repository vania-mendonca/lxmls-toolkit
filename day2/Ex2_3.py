import lxmls.sequences.hmm as hmmc
import lxmls.readers.simple_sequence as ssr

simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)

initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(
    simple.train.seq_list[0])

print "Initial"
print initial_scores

print "Transition"
print transition_scores

print "Final"
print final_scores

print "Emission"
print emission_scores
