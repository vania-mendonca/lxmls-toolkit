import lxmls.sequences.hmm as hmmc
import lxmls.readers.simple_sequence as ssr

simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)

print "Initial Probabilities:", hmm.initial_probs
# Correct

print "Transition Probabilities:", hmm.transition_probs
# Correct

print "Final Probabilities:", hmm.final_probs
# Correct

print "Emission Probabilities", hmm.emission_probs
# Correct
