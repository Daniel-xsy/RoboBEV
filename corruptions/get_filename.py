from petrel_client.client import Client

client = Client('~/petreloss.conf')

def main():
    severities = ['easy', 'mid', 'hard']
    cameras = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

    for severity in severities:
        outpath = f'/mnt/petrelfs/xieshaoyuan/models/RoboBEV/corruptions/nusc-generate-night-{severity}.txt'
        files_all = []
        for camera in cameras:
            files = list(client.list(f's3://llmit/xieshaoyuan/cross_domain/LowLight/{severity}/{camera}'))
            files = [f's3://llmit/xieshaoyuan/cross_domain/LowLight/{severity}/{camera}/' + file for file in files]
            files_all += files
        with open(outpath, 'w') as f:
            for file in files_all:
                f.write(file)
                f.write('\n')


main()