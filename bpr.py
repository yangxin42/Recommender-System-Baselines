# coding = utf-8
import numpy as np
import random
import math


parameter = {
    "path": "./ml-100k.csv.clean",
    "split": " ",
    "learnrate": 0.05,
    "trainratio": 0.8,
    "D": 10,
    "lambda": 0.01,
    "topn": 5,
    "epoch_num": 100,
}


class BPR():

    def __init__(self, parameter):
        self.initialize(parameter)
        self.train()
        self.test()

    def initialize(self, parameter):
        self.trainratio = parameter["trainratio"]
        self.path = parameter["path"]
        self.D = parameter["D"]
        self.learnrate = parameter["learnrate"]
        self._lambda = parameter["lambda"]
        self.topn = parameter["topn"]
        self.epoch_num = parameter["epoch_num"]

        self.load_data(self.path)
        self.uservec = np.random.random((self.usernum + self.beginid, self.D))
        self.itemvec = np.random.random((self.itemnum + self.beginid, self.D))

    def train(self):
        self.iteration = 0
        self.epoch = 0
        print("Start training...")
        while (1):
            if self.epoch > self.epoch_num:
                break
            u, vi, vj = self.sample()

            x = np.dot(self.uservec[u], self.itemvec[vi] - self.itemvec[vj])
            self.uservec[u] += self.learnrate * (
                self.coef(x) * (self.itemvec[vi] - self.itemvec[vj]) - self._lambda * self.uservec[u])
            self.itemvec[vi] += self.learnrate * (self.coef(x) * self.uservec[u] - self._lambda * self.itemvec[vi])
            self.itemvec[vj] += self.learnrate * (self.coef(x) * -1 * self.uservec[u] - self._lambda * self.itemvec[vj])

            self.iteration += 1
            if self.iteration % (self.triplenum / 2) == 0:
                print("epoch: %d, auc:%.6f" % (self.epoch, self.auc()))
                self.epoch += 1
        print("Training completed.")

    def test(self):
        self.recommend = [[] for i in range(self.usernum + self.beginid)]
        for uid in self.testdata.keys():
            if uid not in self.traindata.keys():
                continue
            recommend = []
            for vid in range(self.itemnum):
                if self.beginid:
                    vid += 1
                if vid not in self.traindata[uid].keys():
                    recommend.append((vid, self.predict(uid, vid)))
            recommend = sorted(recommend, key=lambda pair: -pair[1])

            for i in range(recommend.__len__()):
                self.recommend[uid].append( recommend[i])

        p, r = self.precision_recall()
        print("precision@%d: %f,\nrecall@%d: %f," % (self.topn, p, self.topn, r))
        print("ndcg@%d:%f" % (self.topn, self.ndcg()))

    def load_data(self, path):
        print("Loading data form " + self.path + "...",end=" ")
        f = open(path, "r")
        users = []
        items = []
        lines = f.readlines()
        self.triplenum = lines.__len__()
        for line in lines:
            if line == "" or line == None: break
            line = line.split()
            if int(line[0]) not in users:
                users.append(int(line[0]))
            if int(line[1]) not in items:
                items.append(int(line[1]))
        self.usernum = users.__len__()
        self.beginid = min(users)
        self.itemnum = items.__len__()
        self.rating = {}
        self.traindata = {}
        self.testdata = {}
        """
        self.rating/testdata/traindata = {
            userid_i:
                {itemid_j:rate_ij,
                itemid_k:rate_ik,
                }
            ,
            useid_l:[
                {itemid_m:rate_lm},
            ],
        }
        """
        print("completed.\n %d users, %d items, %d triples." % (
        self.usernum, self.itemnum, self.triplenum))
        self.traindatanum = 0
        self.testdatanum = 0
        self.train_item_list = []
        for line in lines:
            line = line.split()
            if line == "" or line == None: break
            coin = random.random()
            self.append2json(line, self.rating)
            if coin <= self.trainratio:
                self.append2json(line, self.traindata)
                if int(line[1]) not in self.train_item_list:
                    self.train_item_list.append(int(line[1]))
                self.traindatanum += 1
            else:
                self.append2json(line, self.testdata)
                self.testdatanum += 1
        print("Split training set : %d, testing set : %d\n" % (self.traindatanum, self.testdatanum))

    def append2json(self, line, dic):
        if int(line[0]) in dic.keys():
            dic[int(line[0])][int(line[1])] = float(line[2])
        else:
            dic[int(line[0])] = {int(line[1]): float(line[2])}

    def sample(self):
        u = vi = vj = -1
        while (1):
            u = int(random.random() * self.usernum)
            if self.beginid == 1:
                u += 1
            if u in self.traindata.keys():
                break

        while (1):
            vi = int(random.random() * self.itemnum)
            if self.beginid == 1:
                vi += 1
            if vi in self.traindata[u]:
                break

        while (1):
            vj = int(random.random() * self.itemnum)
            if self.beginid == 1:
                vj += 1
            if vj not in self.traindata[u]:
                break
        return u, vi, vj

    def coef(self, x):
        if math.fabs(x) > 700:
            x = 700 if x > 0 else -700
        return 1 - 1 / (1 + math.exp(-x))

    def predict(self, u, v):
        return np.dot(self.uservec[u], self.itemvec[v])

    def auc(self):
        sum_auc = 0.0
        for uid in self.testdata.keys():
            auc = 0.0
            M = self.testdata[uid].keys().__len__()
            if uid not in self.traindata.keys():
                continue
            recommend = []
            for vid in range(self.itemnum):
                if self.beginid:
                    vid += 1
                if vid not in self.traindata[uid].keys():
                    recommend.append((vid, self.predict(uid, vid)))
            recommend = sorted(recommend, key=lambda pair: -pair[1])
            N = recommend.__len__() - M
            for i in range(recommend.__len__()):
                if recommend[i][0] in self.testdata[uid].keys():
                    auc += N - i
            auc -= M * (M + 1) / 2
            auc /= (M * N)
            sum_auc += auc
        return sum_auc / self.testdata.keys().__len__()

    def precision_recall(self):
        recallsum = 0
        tp = 0.0
        for uid in self.testdata.keys():
            if uid not in self.traindata.keys():
                continue
            for vid in self.testdata[uid].keys():
                if vid in self.train_item_list:
                    recallsum += 1
            recommend = self.recommend[uid]

            for i in range(self.topn):
                if recommend[i][0] in self.testdata[uid].keys():
                    tp += 1
        return tp/(self.usernum * self.topn), tp / recallsum

    def ndcg(self):
        sum_ndcg = 0.0
        for uid in self.testdata.keys():
            if uid not in self.traindata.keys():
                continue
            recommend = self.recommend[uid]
            l = []
            for (vid, socre) in recommend:
                if vid in self.testdata[uid].keys():
                    l.append(1)
                else:
                    l.append(0)

            tmp = sorted(l, reverse=True)
            max_dcg_at_k = tmp[0]
            for i in range(1, self.topn):
                max_dcg_at_k += tmp[i] / math.log(i + 1, 2)

            if not max_dcg_at_k:
                sum_ndcg += 0
            else:
                dcg_at_k = l[0]
                for i in range(1, self.topn):
                    dcg_at_k += l[i] / math.log(i + 1, 2)
                sum_ndcg += dcg_at_k / max_dcg_at_k

        ndcg = sum_ndcg / self.usernum
        return ndcg


def run():
    bpr = BPR(parameter)


if __name__ == '__main__':
    run()

