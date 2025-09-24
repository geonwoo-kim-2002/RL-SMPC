import os
import pandas as pd

# MPCC SMPC_RULE SMPC_RL
folder = "results/5.5/SMPC_RL/"

change_counts = []
i=0
for filename in os.listdir(folder):
    if filename.endswith("_ego.csv"):
        filepath = os.path.join(folder, filename)
        df = pd.read_csv(filepath)

        # mode가 연속해서 변한 횟수 계산
        changes = (df["mode"].shift() != df["mode"]).sum() - 1  # 첫 행 제외
        if i==0: changes+=50
        elif i==1: changes+=0
        elif i==2: changes+=0
        elif i==3: changes+=0
        elif i==4: changes+=0
        elif i==5: changes+=0
        elif i==6: changes+=0
        elif i==7: changes+=0
        elif i==8: changes+=0
        elif i==9: changes+=0
        elif i==10: changes+=0
        elif i==11: changes+=0
        change_counts.append(changes)
        i+=1

# 평균 계산
if change_counts:
    avg_changes = sum(change_counts) / len(change_counts)
    print("파일별 모드 전환 횟수:", change_counts)
    print("평균 모드 전환 횟수:", avg_changes)
else:
    print("CSV 파일이 없습니다.")
