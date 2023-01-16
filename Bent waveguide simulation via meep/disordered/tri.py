import time
import math
import meep as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#path = "output/"

Lwg = 80 # length of the PhCWG
Lbend = 20
Lstr =(Lwg-Lbend)/2
S = 3 # shift parameter "S", defines the waveguide width and the interface symmetry when Isym is set to 0
##### setting of parameters #####
dpml=1
widthPhC = 20+dpml
lengthPhC=Lwg
lengPhC_z=2*Lstr-Lbend/2
widthPhC_z=(20+Lbend/2)
Lwire = 10
ConnectionWaveguide = Lwire+dpml
r = 0.2572
rt=0.4
pi=np.pi
iud=1
n_eff = 2.65
fcen = 0.3 
f_input=205
check_dBm = 50
a0=400
c_const0=299792458
fcen=1000*f_input*a0/c_const0
df = 0.05
nfreq = 500 # number of frequencies at which to compute flux
resolution = 16
Isym = 0 # interface symmetry 0=free, 2=mirror, 1=glide
wgi = (S+3)/6
if Isym==0 :
    symi = (S+1)/4
elif Isym==1:
    symi = 1/2
elif Isym==2:
    symi = 0
p2hs = 0
Si_w = 1 # silicon wire width
So_w = Si_w #source width
if S==-1:
    p2vs = 0.25 # vertical shift of port 2 for some waveguides like W1/6s&W1/3
    num_del = 10
elif S==-2:
    p2vs = 0.25 
    num_del = 5
    p2hs = 0.5
else:
    p2vs = 0 
    num_del = 0

length = lengthPhC + 2*ConnectionWaveguide
width = widthPhC
Nx = int(lengthPhC)
Ny = int(widthPhC)
eps = n_eff**2
xs=0
ys=0 #modifying initial coordinate of the two PhC regions. Is zero if S=0

#####  

def phc_trans(PhC = True,Zbend = True, lengthPhC = 150, decay_check=0, T_decay=500):
    """
    <変数の説明>
    PhC...PhC(フォトニック決勝)を配置するかどうか。Falseで直線導波路
    lengthPhC...PhC導波方向の長さ
    widthPhC...PhC垂直方向の幅。PMLと被ってるので適当。
    ConnectionWaveguide...PhCに接続するSi導波路(棒の部分)の長さ
    wgi...導波路の幅を調整する。1で丸穴一個分空いてることを意味する。0.7とかにすると狭くなってバンドの形が変わる、っていうのはDaii君の研究とも絡む。
    r...穴の半径。ふつうはa/4くらい。meepだと格子定数は1で固定だから、格子定数との比を入力すればOK
    n_eff...屈折率。2次元だと2.5~2.7くらいにしておくと3次元のSi系(n_Si=3.48)と結果が近くなる。違う材料を使うときは要調整、通常はdefaultで大丈夫。
    fcen...入力光（ガウシアンビーム）の中心周波数。知りたいPhCバンドの周波数近くに設定する
    df...入力光（ガウシアンビーム）の半値幅（で合ってる？）
    nfreq...入力光（ガウシアンビーム）のきめ細かさ
    resolution...メッシュの細かさ。2^nにすると計算が軽くなるらしい。
    T_dacay...反復計算数。小さいと誤差が増え、大きいと時間がかかる。sim.run(until_after_sources=...)で計算時間を見積もってから変えるとよさそう
    decay_check...解の収束をどこで判定するか、位置を指定。defaultでOK

    <備考>
    ・meepでは格子定数aはパラメータに含まれないので設定不要
    　誘電体を使うときは入力するらしい（スケール依存性が出るから）
    ・THzやnmは使用せず、すべて規格化周波数で入力する (周波数はωa/2πcで直す)
    """

    length = lengthPhC + 2*ConnectionWaveguide
    width = widthPhC
    lengthPhC_z=(2*Lstr-Lbend/2)
    widthPhC_z=(20+Lbend/2)
    Nx = int(lengthPhC)
    Ny = int(widthPhC)
    Nxz1=int(Lstr)
    Nxz2=int(Lbend)
    Nxz3=Nxz1
    Nyz = 2*int(min([(lengPhC_z/2+Lbend/4)/np.sqrt(3),np.sqrt(3)*Lbend/2+20]))
    eps = n_eff**2
    xs=S/4
    ys=S/4/np.sqrt(3) #modifying initial coordinate of the two PhC regions. Is zero if S=0


    ##### settings of geometry #####
    if PhC:
        #zbend waveguide
        if Zbend:
            # initialization
            cell = mp.Vector3(Lwg-1.5*Lbend+2*ConnectionWaveguide,(width+Lbend/2)*np.sqrt(3),0)
            blk = mp.Block(mp.Vector3(Lwg-1.5*Lbend,(widthPhC+Lbend/2)*np.sqrt(3),mp.inf),
                                center=mp.Vector3(),
                                material=mp.Medium(epsilon=eps))
            geometry=[blk]

            for j in range(Nyz):
                for i in range(Nxz1+3+int(np.floor(1.5*j))):
                    e11=xs+Lbend/4+3/4-i+1.5*j
                    e12=ys+np.sqrt(3)*Lbend/4 + np.sqrt(3)/4+np.sqrt(3)*j/2
                    vertices_u1=[mp.Vector3(e11+iud*rt*np.cos(pi/6+2*pi/3*0),e12+iud*rt*np.sin(pi/6+2*pi/3*0)),
                                mp.Vector3(e11+iud*rt*np.cos(pi/6+2*pi/3*1),e12+iud*rt*np.sin(pi/6+2*pi/3*1)),
                                mp.Vector3(e11+iud*rt*np.cos(pi/6+2*pi/3*2),e12+iud*rt*np.sin(pi/6+2*pi/3*2))
                                ]

                    vertices_d1=[mp.Vector3(-e11+iud*rt*np.cos(pi/2+2*pi/3*0),-e12+iud*rt*np.sin(pi/2+2*pi/3*0)),
                                mp.Vector3(-e11+iud*rt*np.cos(pi/2+2*pi/3*1),-e12+iud*rt*np.sin(pi/2+2*pi/3*1)),
                                mp.Vector3(-e11+iud*rt*np.cos(pi/2+2*pi/3*2),-e12+iud*rt*np.sin(pi/2+2*pi/3*2))
                                ]
                    geometry.append(mp.Prism(vertices=vertices_u1,height=mp.inf))
                    geometry.append(mp.Prism(vertices=vertices_d1,height=mp.inf))

                for k in range(Nxz2):
                    e21=xs+Lbend/4+3/4-k/2+1.5*j
                    e22=ys+np.sqrt(3)*Lbend/4 + np.sqrt(3)/4-k*np.sqrt(3)/2+np.sqrt(3)*j/2
                    vertices_u2=[mp.Vector3(e21+iud*rt*np.cos(pi/6+2*pi/3*0),e22+iud*rt*np.sin(pi/6+2*pi/3*0)),
                                 mp.Vector3(e21+iud*rt*np.cos(pi/6+2*pi/3*1),e22+iud*rt*np.sin(pi/6+2*pi/3*1)),
                                 mp.Vector3(e21+iud*rt*np.cos(pi/6+2*pi/3*2),e22+iud*rt*np.sin(pi/6+2*pi/3*2))
                                ]

                    vertices_d2=[mp.Vector3(-e21+iud*rt*np.cos(pi/2+2*pi/3*0),-e22+iud*rt*np.sin(pi/2+2*pi/3*0)),
                                 mp.Vector3(-e21+iud*rt*np.cos(pi/2+2*pi/3*1),-e22+iud*rt*np.sin(pi/2+2*pi/3*1)),
                                 mp.Vector3(-e21+iud*rt*np.cos(pi/2+2*pi/3*2),-e22+iud*rt*np.sin(pi/2+2*pi/3*2))
                                ]
                    geometry.append(mp.Prism(vertices=vertices_u2,height=mp.inf))
                    geometry.append(mp.Prism(vertices=vertices_d2,height=mp.inf))
                for l in range(Nxz3):
                    e31=xs-Lbend/4+3/4+l+1.5*j
                    e32=ys-np.sqrt(3)*Lbend/4 + np.sqrt(3)/4+np.sqrt(3)*j/2
                    vertices_u3=[mp.Vector3(e31+iud*rt*np.cos(pi/6+2*pi/3*0),e32+iud*rt*np.sin(pi/6+2*pi/3*0)),
                                 mp.Vector3(e31+iud*rt*np.cos(pi/6+2*pi/3*1),e32+iud*rt*np.sin(pi/6+2*pi/3*1)),
                                 mp.Vector3(e31+iud*rt*np.cos(pi/6+2*pi/3*2),e32+iud*rt*np.sin(pi/6+2*pi/3*2))
                                ]
                    vertices_d3=[mp.Vector3(-e31+iud*rt*np.cos(pi/2+2*pi/3*0),-e32+iud*rt*np.sin(pi/2+2*pi/3*0)),
                                 mp.Vector3(-e31+iud*rt*np.cos(pi/2+2*pi/3*1),-e32+iud*rt*np.sin(pi/2+2*pi/3*1)),
                                 mp.Vector3(-e31+iud*rt*np.cos(pi/2+2*pi/3*2),-e32+iud*rt*np.sin(pi/2+2*pi/3*2))
                                ]
                    geometry.append(mp.Prism(vertices=vertices_u3,height=mp.inf))
                    geometry.append(mp.Prism(vertices=vertices_d3,height=mp.inf))

            if num_del>0:
                for k in range(num_del):
                    e01=xs+Lbend/4+3/4-(Nxz1-k)
                    e02=ys+np.sqrt(3)*Lbend/4 + np.sqrt(3)/4
                    vertices012=[mp.Vector3(e01+iud*rt*np.cos(pi/6+2*pi/3*0),e02+iud*rt*np.sin(pi/6+2*pi/3*0)),
                                 mp.Vector3(e01+iud*rt*np.cos(pi/6+2*pi/3*1),e02+iud*rt*np.sin(pi/6+2*pi/3*1)),
                                 mp.Vector3(e01+iud*rt*np.cos(pi/6+2*pi/3*2),e02+iud*rt*np.sin(pi/6+2*pi/3*2))
                                ]
                    geometry.append(mp.Prism(vertices=vertices012,height=mp.inf,material=mp.Medium(epsilon=eps)))


                    e03=-(xs+Lbend/4+3/4-(Nxz1-k)+p2hs)
                    e04=ys-np.sqrt(3)*Lbend/4 + np.sqrt(3)/4
                    vertices034=[mp.Vector3(e03+iud*rt*np.cos(pi/6+2*pi/3*0),e04+iud*rt*np.sin(pi/6+2*pi/3*0)),
                                 mp.Vector3(e03+iud*rt*np.cos(pi/6+2*pi/3*1),e04+iud*rt*np.sin(pi/6+2*pi/3*1)),
                                 mp.Vector3(e03+iud*rt*np.cos(pi/6+2*pi/3*2),e04+iud*rt*np.sin(pi/6+2*pi/3*2))
                                ]
                    geometry.append(mp.Prism(vertices=vertices034,height=mp.inf,material=mp.Medium(epsilon=eps)))
            # silicon wire
            wire1 = mp.Block(mp.Vector3(ConnectionWaveguide,Si_w*np.sqrt(3),mp.inf),
                                center=mp.Vector3(-(2*Lstr-Lbend/2)/2-ConnectionWaveguide/2,p2vs*np.sqrt(3)+np.sqrt(3)*Lbend/4,0),
                                material=mp.Medium(epsilon=eps))
            geometry.append(wire1)

            wire2 = mp.Block(mp.Vector3(ConnectionWaveguide,Si_w*np.sqrt(3),mp.inf),
                                center=mp.Vector3((2*Lstr-Lbend/2)/2+ConnectionWaveguide/2,p2vs*np.sqrt(3)-np.sqrt(3)*Lbend/4,0),
                                material=mp.Medium(epsilon=eps))
            geometry.append(wire2)



            # Gaussian
            sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                                component=mp.Hz,
                                center=mp.Vector3(-(2*Lstr-Lbend/2)/2-ConnectionWaveguide+dpml+Lwire/2,np.sqrt(3)*Lbend/4+p2vs*np.sqrt(3)),
                                size=mp.Vector3(0,So_w))]


            #end zbend
        else:  #straight waveguide

            # Si waveguide
            cell = mp.Vector3(length,width*np.sqrt(3),0)
            waveguide = mp.Block(mp.Vector3(mp.inf,Si_w*np.sqrt(3),mp.inf),
                            center=mp.Vector3(0,p2vs*np.sqrt(3),0),
                            material=mp.Medium(epsilon=eps))
            geometry = [waveguide]


        
            # slab
            blk = mp.Block(mp.Vector3(lengthPhC,widthPhC*np.sqrt(3),mp.inf),
                                    center=mp.Vector3(),
                                    material=mp.Medium(epsilon=eps))

            geometry.append(blk)
            # arrange air-holes
            shift_y = np.sqrt(3)
            for j in range(Ny):
                for i in range(-1,Nx+2):
                    e11=i-Nx/2+symi
                    e12=wgi*np.sqrt(3)/2 + shift_y*j
                    vertices_u1=[mp.Vector3(e11+iud*rt*np.cos(pi/6+2*pi/3*0),e12+iud*rt*np.sin(pi/6+2*pi/3*0)),
                                mp.Vector3(e11+iud*rt*np.cos(pi/6+2*pi/3*1),e12+iud*rt*np.sin(pi/6+2*pi/3*1)),
                                mp.Vector3(e11+iud*rt*np.cos(pi/6+2*pi/3*2),e12+iud*rt*np.sin(pi/6+2*pi/3*2))
                                ]
                    geometry.append(mp.Prism(vertices=vertices_u1,height=mp.inf))

                    e21=i-Nx/2-symi
                    e22=-(wgi*np.sqrt(3)/2 + shift_y*j)
                    vertices_d1=[mp.Vector3(e21+iud*rt*np.cos(pi/2+2*pi/3*0),e22+iud*rt*np.sin(pi/2+2*pi/3*0)),
                                mp.Vector3(e21+iud*rt*np.cos(pi/2+2*pi/3*1),e22+iud*rt*np.sin(pi/2+2*pi/3*1)),
                                mp.Vector3(e21+iud*rt*np.cos(pi/2+2*pi/3*2),e22+iud*rt*np.sin(pi/2+2*pi/3*2))
                                ]
                    geometry.append(mp.Prism(vertices=vertices_d1,height=mp.inf))

                    e31=i-(Nx+1)/2+symi
                    e32=wgi*np.sqrt(3)/2 + shift_y*(j+1/2)
                    vertices_u2=[mp.Vector3(e31+iud*rt*np.cos(pi/6+2*pi/3*0),e32+iud*rt*np.sin(pi/6+2*pi/3*0)),
                                mp.Vector3(e31+iud*rt*np.cos(pi/6+2*pi/3*1),e32+iud*rt*np.sin(pi/6+2*pi/3*1)),
                                mp.Vector3(e31+iud*rt*np.cos(pi/6+2*pi/3*2),e32+iud*rt*np.sin(pi/6+2*pi/3*2))
                                ]
                    geometry.append(mp.Prism(vertices=vertices_u2,height=mp.inf))

                    e41=i-(Nx+1)/2-symi
                    e42= -(wgi*np.sqrt(3)/2 + shift_y*(j+1/2))
                    vertices_d2=[mp.Vector3(e41+iud*rt*np.cos(pi/2+2*pi/3*0),e42+iud*rt*np.sin(pi/2+2*pi/3*0)),
                                mp.Vector3(e41+iud*rt*np.cos(pi/2+2*pi/3*1),e42+iud*rt*np.sin(pi/2+2*pi/3*1)),
                                mp.Vector3(e41+iud*rt*np.cos(pi/2+2*pi/3*2),e42+iud*rt*np.sin(pi/2+2*pi/3*2))
                                ]
                    geometry.append(mp.Prism(vertices=vertices_d2,height=mp.inf))
                    #geometry.append(mp.Cylinder(r, center=mp.Vector3(i-N/2,-wgi*np.sqrt(3)/2)))
            if num_del>0:
                for k in range(num_del):
                    e01=k-Nx/2+symi
                    e02=wgi*np.sqrt(3)/2 + shift_y*0
                    vertices012=[mp.Vector3(e01+iud*rt*np.cos(pi/6+2*pi/3*0),e02+iud*rt*np.sin(pi/6+2*pi/3*0)),
                                 mp.Vector3(e01+iud*rt*np.cos(pi/6+2*pi/3*1),e02+iud*rt*np.sin(pi/6+2*pi/3*1)),
                                 mp.Vector3(e01+iud*rt*np.cos(pi/6+2*pi/3*2),e02+iud*rt*np.sin(pi/6+2*pi/3*2))
                                ]
                    geometry.append(mp.Prism(vertices=vertices012,height=mp.inf,material=mp.Medium(epsilon=eps)))

                    e03=(Nx-k)-Nx/2+symi
                    e04=wgi*np.sqrt(3)/2 + shift_y*0
                    vertices034=[mp.Vector3(e03+iud*rt*np.cos(pi/6+2*pi/3*0),e04+iud*rt*np.sin(pi/6+2*pi/3*0)),
                                 mp.Vector3(e03+iud*rt*np.cos(pi/6+2*pi/3*1),e04+iud*rt*np.sin(pi/6+2*pi/3*1)),
                                 mp.Vector3(e03+iud*rt*np.cos(pi/6+2*pi/3*2),e04+iud*rt*np.sin(pi/6+2*pi/3*2))
                                ]
                    geometry.append(mp.Prism(vertices=vertices034,height=mp.inf,material=mp.Medium(epsilon=eps)))
            # silicon wire
            wire1 = mp.Block(mp.Vector3(ConnectionWaveguide,Si_w*np.sqrt(3),mp.inf),
                                center=mp.Vector3(-lengthPhC/2-ConnectionWaveguide/2,p2vs*np.sqrt(3),0),
                                material=mp.Medium(epsilon=eps))
            geometry.append(wire1)

            wire2 = mp.Block(mp.Vector3(ConnectionWaveguide,Si_w*np.sqrt(3),mp.inf),
                                center=mp.Vector3(lengthPhC/2+ConnectionWaveguide/2,p2vs*np.sqrt(3),0),
                                material=mp.Medium(epsilon=eps))
            geometry.append(wire2)
            
            
            wire3 = mp.Block(mp.Vector3(ConnectionWaveguide,Si_w*np.sqrt(3),mp.inf),
                                center=mp.Vector3(lengthPhC/2+ConnectionWaveguide/2,p2vs*np.sqrt(3),0))

            wire4 = mp.Block(mp.Vector3(ConnectionWaveguide,Si_w*np.sqrt(3),mp.inf),
                                center=mp.Vector3(lengthPhC/2+ConnectionWaveguide/2,0*np.sqrt(3),0),
                                material=mp.Medium(epsilon=eps))

            if num_del>0:
                geometry.append(wire3)
                geometry.append(wire4) 



            # Gaussian
            sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                            component=mp.Hz,
                            center=mp.Vector3(-length/2 +dpml+Lwire/2,p2vs*np.sqrt(3)),
                            size=mp.Vector3(0,So_w))]
    else:
        cell = mp.Vector3(length,width*np.sqrt(3),0)

        # Si waveguide
        waveguide = mp.Block(mp.Vector3(mp.inf,Si_w*np.sqrt(3),mp.inf),
                         center=mp.Vector3(0,p2vs*np.sqrt(3),0),
                         material=mp.Medium(epsilon=eps))
        geometry = [waveguide]
        # Gaussian
        sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                         component=mp.Hz,
                         center=mp.Vector3(-length/2 +dpml+Lwire/2,p2vs*np.sqrt(3)),
                         size=mp.Vector3(0,So_w))]




        # z-symmetry (上下対称なら計算が軽くなる。対称性が無いなら消す)
        # sym = [mp.Mirror(mp.Y, phase=-1)]
    
    # PML
    pml_layers = [mp.PML(dpml)]
    ####
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        # symmetries=sym,
                        resolution=resolution)

    if Zbend:
        #tran_in = mp.FluxRegion(center=mp.Vector3(-lengthPhC/2-1,0),size=mp.Vector3(0, 2*wgi))
        tran_out = mp.FluxRegion(center=mp.Vector3((2*Lstr-Lbend/2)/2+ConnectionWaveguide-dpml-Lwire/2,0*np.sqrt(3)-np.sqrt(3)*Lbend/4),size=mp.Vector3(0, 1.5*Si_w))
        #trans_in = sim.add_flux(fcen, df, nfreq, tran_in)
        trans_out = sim.add_flux(fcen, df, nfreq, tran_out)


        tran_inPC1 = mp.FluxRegion(center=mp.Vector3(-(2*Lstr-Lbend/2)/2+15,0*np.sqrt(3)+np.sqrt(3)*Lbend/4),size=mp.Vector3(0, 6))
        trans_inPC1 = sim.add_flux(fcen, df, nfreq, tran_inPC1)
        tran_inPC2 = mp.FluxRegion(center=mp.Vector3((2*Lstr-Lbend/2)/2-15,0*np.sqrt(3)-np.sqrt(3)*Lbend/4),size=mp.Vector3(0, 6))
        trans_inPC2 = sim.add_flux(fcen, df, nfreq, tran_inPC2)
    else:
        #tran_in = mp.FluxRegion(center=mp.Vector3(-lengthPhC/2-1,0),size=mp.Vector3(0, 2*wgi))
        tran_out = mp.FluxRegion(center=mp.Vector3(length/2-dpml-Lwire/2,0*np.sqrt(3)),size=mp.Vector3(0, 1.5*Si_w))
        #trans_in = sim.add_flux(fcen, df, nfreq, tran_in)
        trans_out = sim.add_flux(fcen, df, nfreq, tran_out)


        tran_inPC1 = mp.FluxRegion(center=mp.Vector3(-lengthPhC/2+15,0*np.sqrt(3)),size=mp.Vector3(0, 6))
        trans_inPC1 = sim.add_flux(fcen, df, nfreq, tran_inPC1)
        tran_inPC2 = mp.FluxRegion(center=mp.Vector3(lengthPhC/2-15,0*np.sqrt(3)),size=mp.Vector3(0, 6))
        trans_inPC2 = sim.add_flux(fcen, df, nfreq, tran_inPC2)

 
    """# show geometry
    %matplotlib inline
    f = plt.figure(dpi=150)
    sim.plot2D(ax=f.gca())
    plt.show()   """ 







    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Hz, mp.Vector3(decay_check), pow(10,-check_dBm/10)))
    #sim.run(until=T_decay)

    freqs = mp.get_flux_freqs(trans_out)
    psd_inPC1 = mp.get_fluxes(trans_inPC1)
    psd_inPC2 = mp.get_fluxes(trans_inPC2)
    psd_out = mp.get_fluxes(trans_out)


    return freqs, psd_inPC1, psd_inPC2, psd_out



if __name__ == "__main__":  
    time_start = time.perf_counter()

    a = a0
    c_const = c_const0
    freqs_wo, psd_inPC1_wo, psd_inPC2_wo,psd_out_wo = phc_trans(PhC = False, Zbend=False,lengthPhC = Lwg, decay_check=20, T_decay=1000)
    freqs_s,  psd_inPC1_s,  psd_inPC2_s, psd_out_s  = phc_trans(PhC = True, Zbend=False,lengthPhC = Lwg, decay_check=20, T_decay=3000)
    freqs_z,  psd_inPC1_z,  psd_inPC2_z, psd_out_z  = phc_trans(PhC = True, Zbend=True,lengthPhC = Lwg, decay_check=20, T_decay=3000)

    freqs=c_const * np.array(freqs_s)/a/1000


    df = pd.DataFrame()
    #df["normalized_frequency"] = np.array(freqs_w)
    df["freq"] = freqs
    df["T_s"] = np.array(psd_out_s)/np.array(psd_out_wo)
    df["T_z"] = np.array(psd_out_z)/np.array(psd_out_wo)
    df.to_csv("transmittance_S="+str(S)+"_f="+str(f_input)+"_"+str(check_dBm)+"dBm"+".csv", index=False)

    
    plt.plot(freqs, np.array(psd_out_s)/np.array(psd_out_wo))
    plt.plot(freqs, np.array(psd_out_z)/np.array(psd_out_wo))
    plt.xlabel("Frequency[THz]",fontsize=20)
    plt.ylabel("Transmittance",fontsize=20)
    plt.xlim([195,235])
    plt.ylim([0,1.2])
    #plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("transmittance_S="+str(S)+"_f="+str(f_input)+"_"+str(check_dBm)+"dBm"+".png")
    
    time_end = time.perf_counter()
    time = time_end - time_start
    print("The necessary time: {:.3f}s".format(time))

    f = open('totaltime1.txt', 'a')
    f.write("Lwg="+str(Lwg)+"_S="+str(S)+"_"+str(check_dBm)+"dBm_"+"The necessary time: {:.3f}s".format(time))
    f.close()