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
    "topn": 5,
}


class POP():

    def __init__(self, parameter):
        self.initzlize(parameter)
        self.train()
        self.test()

    def initzlize(self, parameter):
        self.trainratio = parameter["trainratio"]
        self.path = parameter["path"]
        self.D = parameter["D"]
        self.topn = parameter["topn"]
        self.leatnrate = parameter["learnrate"]
        self.load_data(self.path)

    def train(self):
        print("Start training...")
        items = []
        self.item_count = {}
        for uid in self.traindata.keys():
            for vid in self.traindata[uid]:
                if vid not in items:
                    items.append(vid)
                    self.item_count[vid] = 1
                else:
                    self.item_count[vid] += 1
        self.item_count_list = []
        for item, count in self.item_count.items():
            self.item_count_list.append((item, count))
        self.item_count_list = sorted(self.item_count_list, key = lambda pair: -pair[1])

        print("Training completed.")


    def test(self):
        self.recommend = []
        for item,count in self.item_count_list:
            self.recommend.append(item)

        print("auc: %f"%self.auc())
        p,r = self.precision_recall()
        print("precision@%d: %f\nrecall@%d: %f," % (
        self.topn, p, self.topn, r))
        print("ndcg@%d:%f" % (self.topn, self.ndcg()))
        print("\n")

    def append2json(self, line, dic):
        if int(line[0]) in dic.keys():
            dic[int(line[0])][int(line[1])] = float(line[2])
        else:
            dic[int(line[0])] = {int(line[1]): float(line[2])}

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
        self.itemnum = items.__len__()
        self.beginid = min(users)
        print("completed.\n %d users, %d items, %d triples." % (
        self.usernum, self.itemnum, self.triplenum))
        self.rating = {}
        self.traindata = {}
        self.testdata = {}
        """
        self.rating = {
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
        self.traindatanum = 0
        self.testdatanum = 0
        self.train_item_list = []
        for line in lines:
            line = line.split()
            if line == "" or line == None: break
            coin = random.random()
            self.append2json(line, self.rating)
            self.triplenum += 1
            if coin <= self.trainratio:
                self.append2json(line, self.traindata)
                if int(line[1]) not in self.train_item_list:
                    self.train_item_list.append(int(line[1]))
                self.traindatanum += 1
            else:
                self.append2json(line, self.testdata)
                self.testdatanum += 1
        print("Split training set : %d, testing set : %d\n" % (self.traindatanum, self.testdatanum))

    def auc(self):
        sum_auc = 0.0
        for uid in self.testdata.keys():
            auc = 0.0
            M = self.testdata[uid].__len__()
            if uid not in self.traindata.keys():
                continue
            recommend = self.item_count_list

            N = recommend.__len__() - M
            for i in range(recommend.__len__()):
                if recommend[i][0] in self.testdata[uid]:
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
            recommend = self.recommend

            for i in range(self.topn):
                if recommend[i] in self.testdata[uid].keys():
                    tp += 1
        return tp/(self.usernum * self.topn), tp / recallsum

    def ndcg(self):
        sum_ndcg = 0.0
        for uid in self.testdata.keys():
            if uid not in self.traindata.keys():
                continue
            recommend = self.item_count_list
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
    pop = POP(parameter)

if __name__ == '__main__':
    run()



