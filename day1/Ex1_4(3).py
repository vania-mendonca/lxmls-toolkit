import lxmls.classifiers.max_ent_online as meoc
import lxmls.readers.sentiment_reader as srs

scr = srs.SentimentCorpus("books")

me_sgd = meoc.MaxEntOnline()

me_sgd.regularizer = 1.0

params_meo_sc = me_sgd.train(scr.train_X, scr.train_y)
y_pred_train = me_sgd.test(scr.train_X, params_meo_sc)

acc_train = me_sgd.evaluate(scr.train_y, y_pred_train)
y_pred_test = me_sgd.test(scr.test_X, params_meo_sc)

acc_test = me_sgd.evaluate(scr.test_y, y_pred_test)

print "Max-Ent Online Amazon Sentiment Accuracy train: %f test: %f" % (
    acc_train, acc_test)

# On Amazon dataset
# train: 0.858750 test: 0.790000
