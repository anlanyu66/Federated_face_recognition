import numpy as np
from matplotlib import pyplot as plt



trainacc_1_worker = np.load('./alexnet/trainacc_1_worker.npy')
trainloss_1_worker = np.load('./alexnet/trainloss_1_worker.npy')

trainloss_3_worker_set0 = np.load('./alexnet/trainloss_3_woker_set_0_diff_5inner_avg.npy')
trainloss_3_worker_set1 = np.load('./alexnet/trainloss_3_woker_set_1_diff_5inner_avg.npy')
trainloss_3_worker_set2 = np.load('./alexnet/trainloss_3_woker_set_2_diff_5inner_avg.npy')

# trainacc_3_worker_set0 = np.load('./alexnet/trainacc_3_worker_set_0.npy')
# trainacc_3_worker_set1 = np.load('./alexnet/trainacc_3_worker_set_1.npy')
# trainacc_3_worker_set2 = np.load('./alexnet/trainacc_3_worker_set_2.npy')

trainacc_3_worker_set0 = np.load('./alexnet/trainacc_3_worker_set_0_diff.npy')
trainacc_3_worker_set1 = np.load('./alexnet/trainacc_3_worker_set_1_diff.npy')
trainacc_3_worker_set2 = np.load('./alexnet/trainacc_3_worker_set_2_diff.npy')

# trainacc_3_worker_set0 = np.load('./alexnet/trainacc_3_worker_set_0_diff_5inner.npy')
# trainacc_3_worker_set1 = np.load('./alexnet/trainacc_3_worker_set_1_diff_5inner.npy')
# trainacc_3_worker_set2 = np.load('./alexnet/trainacc_3_worker_set_2_diff_5inner.npy')

trainacc_3_worker_set0_avg = np.load('./alexnet/trainacc_3_worker_set_0_diff_5inner_avg.npy')
trainacc_3_worker_set1_avg = np.load('./alexnet/trainacc_3_worker_set_1_diff_5inner_avg.npy')
trainacc_3_worker_set2_avg = np.load('./alexnet/trainacc_3_worker_set_2_diff_5inner_avg.npy')

testacc_1_worker = np.load('./alexnet/testacc_1_worker.npy')
testloss_1_worker = np.load('./alexnet/testloss_1_worker.npy')

# testacc_3_worker = np.load('./alexnet/testacc_3_worker.npy')
testacc_3_worker = np.load('./alexnet/testacc_3_worker_diff.npy')
# testacc_3_worker = np.load('./alexnet/testacc_3_worker_diff_5inner.npy')
testacc_3_worker_avg = np.load('./alexnet/testacc_3_worker_diff_5inner_avg.npy')

testloss_3_worker = np.load('./alexnet/testloss_3_worker_diff_5inner_avg.npy')

x = range(100)

# plt.plot(x, trainacc_1_worker, label='train, 1 worker')
# plt.plot(x, trainacc_3_worker_set0, label='train, 3 worker, set 1')
# plt.plot(x, trainacc_3_worker_set1, label='train, 3 worker, set 2')
# plt.plot(x, trainacc_3_worker_set2, label='trian, 3 worker, set 3')
plt.plot(x, trainloss_1_worker, label='train, 1 worker')
plt.plot(x, trainloss_3_worker_set0, label='train, 3 worker, set 1, average')
plt.plot(x, trainloss_3_worker_set1, label='train, 3 worker, set 2, average')
plt.plot(x, trainloss_3_worker_set2, label='train, 3 worker, set 3, average')

# plt.plot(x, trainacc_3_worker_set0_avg, label='train, 3 worker, set 1, average')
# plt.plot(x, trainacc_3_worker_set1_avg, label='train, 3 worker, set 2, average')
# plt.plot(x, trainacc_3_worker_set2_avg, label='trian, 3 worker, set 3, average')

# plt.plot(x, testacc_1_worker, label='test, 1 worker')
# plt.plot(x, testacc_3_worker, label='test, 3 worker')
# plt.plot(x, testacc_3_worker_avg, label='test, 3 worker, average')
plt.plot(x, testloss_1_worker, label='test, 1 worker')
plt.plot(x, testloss_3_worker, label='test, 3 worker, average')

plt.xlabel('epoch')
plt.ylabel('loss')
# plt.ylabel('accuracy')
plt.title('Federated learning vs single model learning (alexnet)')

plt.legend()
plt.savefig('./alexnet/alexnet_diff_identities_loss')
plt.show()
