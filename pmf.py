# coding = utf-8
import numpy as np
import random
import math

parameter = {
    "path":"./ml-100k.csv.clean",
    "split":" ",
    "learnrate":0.005,
    "trainratio":0.8,
    "D":10,
    "epoch_num": 10,
    "topn":5,
    "lambda_u":0.9,
    "lambda_v": 0.9,
}


class PMF():

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
        self.lambda_u = parameter["lambda_u"]
        self.lambda_v = parameter["lambda_v"]
        self.epoch_num = parameter["epoch_num"]
        self.load_data(self.path)
        self.uservec = np.random.random((self.usernum + self.beginid, self.D))
        self.itemvec = np.random.random((self.itemnum + self.beginid, self.D))
        self.indicator = np.zeros((self.usernum + self.beginid, self.itemnum + self.beginid))
        for uid in self.rating.keys():
            for vid in self.rating[uid]:
                self.indicator[uid][vid] = 1

    def train(self):
        self.iteration = 0
        self.epoch = 0
        print("Start training...")
        while (1):
            if self.epoch > self.epoch_num:
                break
            u, v = self.sampleob()
            r = int(self.traindata[u][v])

            self.uservec[u] -= self.leatnrate * (
                self.indicator[u][v] * (-r * self.itemvec[v] + np.linalg.norm(self.itemvec[v]) * self.uservec[
                    u]) + self.lambda_u * np.linalg.norm(self.itemvec[v]) * self.uservec[u])
            self.itemvec[v] -= self.leatnrate * (
                self.indicator[u][v] * (-r * self.uservec[u] + np.linalg.norm(self.uservec[u]) * self.itemvec[
                    v]) + self.lambda_v * np.linalg.norm(self.uservec[u]) * self.itemvec[v])

            self.iteration += 1
            if self.iteration % (self.triplenum/2) == 0:
                self.epoch += 1
                print("epoch: %d, rmse: %.6f" % (self.epoch, self.rmse()))
        print("Training completed.")

    def test(self):
        print("rmse:%f"%(self.rmse(kind="test")))

    def append2json(self,line,dic):
        if int(line[0]) in dic.keys():
            dic[int(line[0])][int(line[1])] = float(line[2])
        else:
            dic[int(line[0])] = {int(line[1]): float(line[2])}

    def load_data(self,path):
        print("Loading data form " + self.path + "...",end=" ")
        f = open(path,"r")
        users = []
        items = []
        lines = f.readlines()
        self.triplenum = lines.__len__()
        for line in lines:
            if line =="" or line ==None:break
            line = line.split()
            if int(line[0]) not in users:
                users.append(int(line[0]))
            if int(line[1]) not in items:
                items.append(int(line[1]))
        self.usernum = users.__len__()
        self.itemnum = items.__len__()
        self.beginid = min(users)
        print("completed.\n %d users, %d items, %d triples."%(self.usernum,self.itemnum,self.triplenum))
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

        for line in lines:
            line = line.split()
            if line =="" or line ==None:break
            coin = random.random()
            self.append2json(line,self.rating)
            self.triplenum +=1
            if coin<=self.trainratio:
                self.append2json(line,self.traindata)
                self.traindatanum += 1
            else:
                self.append2json(line,self.testdata)
                self.testdatanum += 1

        print("Split training set : %d, testing set : %d\n"%(self.traindatanum,self.testdatanum))

    def sampleob(self):
        uid = vid =-1
        while(1):
            uid = int(random.random()*self.usernum)
            if self.beginid == 1:
                uid += 1
            if uid in self.traindata.keys(): break

        while(1):
            vid = int(random.random()*self.itemnum)
            if self.beginid == 1:
                vid += 1
            if vid in self.traindata[uid].keys():
                break
        return uid,vid

    def predict(self,uid,vid):
        return np.dot(self.uservec[uid],self.itemvec[vid])

    def rmse(self,kind = "train"):
        r = 0.0
        count = 0
        if kind == "test":
            for uid in self.testdata.keys():
                if uid not in self.traindata.keys():
                    continue
                for vid in self.testdata[uid]:
                    count += 1
                    r += (self.predict(uid, vid)- self.testdata[uid][vid])*(self.predict(uid, vid) - self.testdata[uid][vid])
            r = r/count
        elif kind == "train":
            for uid in self.traindata.keys():
                for vid in self.traindata[uid]:
                    count += 1
                    r += (self.predict(uid, vid)- self.traindata[uid][vid])*(self.predict(uid, vid) - self.traindata[uid][vid])
            r = r/count
        return math.sqrt(r)


def run():
    pmf = PMF(parameter)

if __name__ == '__main__':
    run()
