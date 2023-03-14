import mmcv


def main():
    pass

if __name__=='__main__':
    filename = '/nvme/konglingdong/models/RoboDet/data/uda/nuscenes_infos_sing_train.pkl'
    data = mmcv.load(filename)
    data['metadata']['domain'] = 'city2city-sing'
    mmcv.dump(data, filename)
