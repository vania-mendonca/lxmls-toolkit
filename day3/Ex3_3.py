import lxmls.readers.pos_corpus as pcc
import lxmls.sequences.id_feature as idfc
import lxmls.sequences.structured_perceptron as spc

data_path = "./data"

# Load the corpus
corpus = pcc.PostagCorpus()

# Load the training, test and development sequences
train_seq = corpus.read_sequence_list_conll(data_path + "/train-02-21.conll",
                                            max_sent_len=10, max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll(data_path + "/test-23.conll",
                                           max_sent_len=10, max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll(data_path + "/dev-22.conll",
                                          max_sent_len=10, max_nr_sent=1000)

feature_mapper = idfc.IDFeatures(train_seq)
feature_mapper.build_features()
print "Perceptron Exercise"

sp = spc.StructuredPerceptron(corpus.word_dict, corpus.tag_dict, feature_mapper)
sp.num_epochs = 20
sp.train_supervised(train_seq)

pred_train = sp.viterbi_decode_corpus(train_seq)
pred_dev = sp.viterbi_decode_corpus(dev_seq)
pred_test = sp.viterbi_decode_corpus(test_seq)

eval_train = sp.evaluate_corpus(train_seq, pred_train)
eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
eval_test = sp.evaluate_corpus(test_seq, pred_test)

print "Structured Perceptron - ID Features Accuracy Train: %.3f Dev: %.3f Test: %.3f" % (
eval_train, eval_dev, eval_test)
