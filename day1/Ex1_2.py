# import lxmls.readers.simple_data_set as sds
import lxmls.readers.sentiment_reader as srs
import lxmls.classifiers.perceptron as percc

# sd = sds.SimpleDataSet(nr_examples=100, g1=[[-1, -1], 1], g2=[[1, 1], 1],
# balance=0.5, split=[0.5, 0, 0.5])
sd = srs.SentimentCorpus("books")

perc = percc.Perceptron()

params_perc_sd = perc.train(sd.train_X, sd.train_y)

y_pred_train = perc.test(sd.train_X, params_perc_sd)
acc_train = perc.evaluate(sd.train_y, y_pred_train)

y_pred_test = perc.test(sd.test_X, params_perc_sd)
acc_test = perc.evaluate(sd.test_y, y_pred_test)

print "Perceptron Simple Dataset Accuracy train: %f test: %f" % (
    acc_train, acc_test)

# fig, axis = sd.plot_data()
# fig, axis = sd.add_line(fig, axis, params_perc_sd, "Perceptron", "blue")
# fig.show()

# On simple dataset
# train: 0.820000 test: 0.960000, train: 0.880000 test: 0.980000

# On Amazon dataset
# train: 0.998750 test: 0.825000
