%% 原始数据的滤波、标准化和整合

% 使用巴特沃斯滤波器进行滤波
Fs = 10;
wp=1.6*2/Fs; %通带边界频率 ，单位为rad/s
ws=3*2/Fs; %阻带边界频率 ，单位为rad/s
Rp=1;           %通带最大波纹度 ，单位dB (不要太小)
Rs=30;          %表示阻带最小衰减，单位dB
[N,Wn]=buttord(wp,ws,Rp,Rs);
[B,A]=butter(N,Wn);

% 健康对照组的数据读取
MainPath = '.\原始数据\健康对照组\XJTU_NIRS_Case\';
PathFileFormation = dir(MainPath);
PathNumber = numel(PathFileFormation);
%记录数据
HC = cell(PathNumber-2, 5);

% 开始每个人的循环
for LoopPathNumer = 3:PathNumber
    filePath = [fullfile(MainPath,PathFileFormation(LoopPathNumer).name) '\'];
    load(strcat(filePath,'syn.txt'));
    %打开文件的顺序，也是记录文件的顺序
    l_h = fread(fopen(strcat(filePath,'l_hb_f.dat')),inf,'float32');
    l_o2 = fread(fopen(strcat(filePath,'l_hbo2_f.dat')),inf,'float32');
    r_h = fread(fopen(strcat(filePath,'r_hb_f.dat')),inf,'float32');
    r_o2 = fread(fopen(strcat(filePath,'r_hbo2_f.dat')),inf,'float32');
    fclose('all');
    
    % 标准化
    l_h = (l_h-mean(l_h))/std(l_h);
    l_o2 = (l_o2-mean(l_o2))/std(l_o2);
    r_h = (r_h-mean(r_h))/std(r_h);
    r_o2 = (r_o2-mean(r_o2))/std(r_o2);
    
    % 零相位滤波
    l_h = filtfilt(B,A,l_h);
    l_o2 = filtfilt(B,A,l_o2);
    r_h = filtfilt(B,A,r_h);
    r_o2 = filtfilt(B,A,r_o2);
    
    HC{LoopPathNumer-2, 1} = l_h;
    HC{LoopPathNumer-2, 2} = l_o2;
    HC{LoopPathNumer-2, 3} = r_h;
    HC{LoopPathNumer-2, 4} = r_o2;
    HC{LoopPathNumer-2, 5} = syn;
    
end

% 抑郁症患者组的数据读取
MainPath = '.\原始数据\抑郁症患者组\XJTU_NIRS_Case\';
PathFileFormation = dir(MainPath);
PathNumber = numel(PathFileFormation);
%记录数据
MDD = cell(PathNumber-2, 5);

% 开始每个人的循环
for LoopPathNumer = 3:PathNumber
    filePath = [fullfile(MainPath,PathFileFormation(LoopPathNumer).name) '\'];
    load(strcat(filePath,'syn.txt'));
    %打开文件的顺序，也是记录文件的顺序
    l_h = fread(fopen(strcat(filePath,'l_hb_f.dat')),inf,'float32');
    l_o2 = fread(fopen(strcat(filePath,'l_hbo2_f.dat')),inf,'float32');
    r_h = fread(fopen(strcat(filePath,'r_hb_f.dat')),inf,'float32');
    r_o2 = fread(fopen(strcat(filePath,'r_hbo2_f.dat')),inf,'float32');
    fclose('all');
    
    % 标准化
    l_h = (l_h-mean(l_h))/std(l_h);
    l_o2 = (l_o2-mean(l_o2))/std(l_o2);
    r_h = (r_h-mean(r_h))/std(r_h);
    r_o2 = (r_o2-mean(r_o2))/std(r_o2);
    
    % 零相位滤波
    l_h = filtfilt(B,A,l_h);
    l_o2 = filtfilt(B,A,l_o2);
    r_h = filtfilt(B,A,r_h);
    r_o2 = filtfilt(B,A,r_o2);
    
    MDD{LoopPathNumer-2, 1} = l_h;
    MDD{LoopPathNumer-2, 2} = l_o2;
    MDD{LoopPathNumer-2, 3} = r_h;
    MDD{LoopPathNumer-2, 4} = r_o2;
    MDD{LoopPathNumer-2, 5} = syn;
    
end

path = 'data.mat';
save(path, 'HC', 'MDD')
