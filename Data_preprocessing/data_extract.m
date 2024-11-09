%% ԭʼ���ݵ��˲�����׼��������

% ʹ�ð�����˹�˲��������˲�
Fs = 10;
wp=1.6*2/Fs; %ͨ���߽�Ƶ�� ����λΪrad/s
ws=3*2/Fs; %����߽�Ƶ�� ����λΪrad/s
Rp=1;           %ͨ������ƶ� ����λdB (��Ҫ̫С)
Rs=30;          %��ʾ�����С˥������λdB
[N,Wn]=buttord(wp,ws,Rp,Rs);
[B,A]=butter(N,Wn);

% ��������������ݶ�ȡ
MainPath = '.\ԭʼ����\����������\XJTU_NIRS_Case\';
PathFileFormation = dir(MainPath);
PathNumber = numel(PathFileFormation);
%��¼����
HC = cell(PathNumber-2, 5);

% ��ʼÿ���˵�ѭ��
for LoopPathNumer = 3:PathNumber
    filePath = [fullfile(MainPath,PathFileFormation(LoopPathNumer).name) '\'];
    load(strcat(filePath,'syn.txt'));
    %���ļ���˳��Ҳ�Ǽ�¼�ļ���˳��
    l_h = fread(fopen(strcat(filePath,'l_hb_f.dat')),inf,'float32');
    l_o2 = fread(fopen(strcat(filePath,'l_hbo2_f.dat')),inf,'float32');
    r_h = fread(fopen(strcat(filePath,'r_hb_f.dat')),inf,'float32');
    r_o2 = fread(fopen(strcat(filePath,'r_hbo2_f.dat')),inf,'float32');
    fclose('all');
    
    % ��׼��
    l_h = (l_h-mean(l_h))/std(l_h);
    l_o2 = (l_o2-mean(l_o2))/std(l_o2);
    r_h = (r_h-mean(r_h))/std(r_h);
    r_o2 = (r_o2-mean(r_o2))/std(r_o2);
    
    % ����λ�˲�
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

% ����֢����������ݶ�ȡ
MainPath = '.\ԭʼ����\����֢������\XJTU_NIRS_Case\';
PathFileFormation = dir(MainPath);
PathNumber = numel(PathFileFormation);
%��¼����
MDD = cell(PathNumber-2, 5);

% ��ʼÿ���˵�ѭ��
for LoopPathNumer = 3:PathNumber
    filePath = [fullfile(MainPath,PathFileFormation(LoopPathNumer).name) '\'];
    load(strcat(filePath,'syn.txt'));
    %���ļ���˳��Ҳ�Ǽ�¼�ļ���˳��
    l_h = fread(fopen(strcat(filePath,'l_hb_f.dat')),inf,'float32');
    l_o2 = fread(fopen(strcat(filePath,'l_hbo2_f.dat')),inf,'float32');
    r_h = fread(fopen(strcat(filePath,'r_hb_f.dat')),inf,'float32');
    r_o2 = fread(fopen(strcat(filePath,'r_hbo2_f.dat')),inf,'float32');
    fclose('all');
    
    % ��׼��
    l_h = (l_h-mean(l_h))/std(l_h);
    l_o2 = (l_o2-mean(l_o2))/std(l_o2);
    r_h = (r_h-mean(r_h))/std(r_h);
    r_o2 = (r_o2-mean(r_o2))/std(r_o2);
    
    % ����λ�˲�
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
