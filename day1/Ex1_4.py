import matplotlib.pyplot as plt
import lxmls.classifiers.max_ent_batch as mebc
import lxmls.readers.simple_data_set as sds
# import lxmls.readers.sentiment_reader as srs

sd = sds.SimpleDataSet(nr_examples=100, g1=[[-1, -1], 1], g2=[[1, 1], 1],
                       balance=0.5, split=[0.5, 0, 0.5])
# sd = srs.SentimentCorpus("books")

me_lbfgs = mebc.MaxEntBatch()
me_lbfgs.regularizer = 1.0 # lambda

params_meb_sd = me_lbfgs.train(sd.train_X, sd.train_y)
y_pred_train = me_lbfgs.test(sd.train_X, params_meb_sd)
acc_train = me_lbfgs.evaluate(sd.train_y, y_pred_train)

y_pred_test = me_lbfgs.test(sd.test_X, params_meb_sd)
acc_test = me_lbfgs.evaluate(sd.test_y, y_pred_test)

print "Max-Ent batch Simple Dataset Accuracy train: %f test: %f" % (
    acc_train, acc_test)
# print "Max-Ent batch Amazon Accuracy train: %f test: %f" % (acc_train, acc_test)

# fig, axis = sd.plot_data()
# fig, axis = sd.add_line(fig, axis, params_meb_sd, "Max-Ent-Batch", "orange")
# plt.show()

# On simple dataset
# train: 0.840000 test: 0.940000
# On Amazon dataset
# train: 0.858125 test: 0.790000
