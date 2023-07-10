import numpy as np

def output_cg(pos,sys):
    print("output: cg.pdb")
    box = sys[3]
    name = "ru"
    resname = "Pl"
    occupency = 1
    temper_f = 0
    element = "P"
    sig = sys[1]
    res = sys[2]

    path = "./cg.pdb"
    f = open(path, 'w')
    f.write("Title   Create by Ambrose from M^5\n")
    f.write(f"CRYST1{box[0]:9.3f}{box[1]:9.3f}{box[2]:9.3f}{box[3]:7.2f}{box[4]:7.2f}{box[5]:7.2f} P 1           1\n")

    for i in range(sig*res):
        if (i==0 or i==1 or i==2):
            resname = "CPm"
            f.write(f"ATOM  {i+1:5} {name+str(i+1):>4} {resname:>3} A{1:>4}    {pos[i,0]:>8.3f}{pos[i,1]:>8.3f}{pos[i,2]:>8.3f}{occupency:6.2f}{temper_f:6.2f}           {element:2}\n")    
        
        elif(i==sig*res-1 or i ==sig*res-2 or i==sig*res-3):
            resname = "CPm"
            f.write(f"ATOM  {i+1:5} {name+str(i+1-(sig*res-3)):>4} {resname:>3} A{int(sig*res/3):>4}    {pos[i,0]:>8.3f}{pos[i,1]:>8.3f}{pos[i,2]:>8.3f}{occupency:6.2f}{temper_f:6.2f}           {element:2}\n")    
        else:
            if((i)%3==0):
                resname = "Pm"
                f.write(f"ATOM  {i+1:5} {name+str(1+(i)%3):>4} {resname:>3} A{int(i/3)+1:>4}    {pos[i,0]:>8.3f}{pos[i,1]:>8.3f}{pos[i,2]:>8.3f}{occupency:6.2f}{temper_f:6.2f}           {element:2}\n")    
            elif((i)%3==1):
                resname = "Pm"
                f.write(f"ATOM  {i+1:5} {name+str(1+(i)%3):>4} {resname:>3} A{int(i/3)+1:>4}    {pos[i,0]:>8.3f}{pos[i,1]:>8.3f}{pos[i,2]:>8.3f}{occupency:6.2f}{temper_f:6.2f}           {element:2}\n")    
            elif((i)%3==2):
                resname = "Pm"
                f.write(f"ATOM  {i+1:5} {name+str(1+(i)%3):>4} {resname:>3} A{int(i/3)+1:>4}    {pos[i,0]:>8.3f}{pos[i,1]:>8.3f}{pos[i,2]:>8.3f}{occupency:6.2f}{temper_f:6.2f}           {element:2}\n")    
    f.write("TER\n")

    for j in range(0,sig):
        for i in range(0,res-1):
            #bond_l.append([(i+1+j*(60)),(i+2+j*(60))])
            f.write(f"CONECT{i+1+j*(res):5}{i+2+j*(res):5}\n")

    f.write("END")

    f.close()
