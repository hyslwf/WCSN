%% ���˲����ݽ���С���任�����ݽض�

% ��ȡ�˲�����
load('data.mat');

% �洢·��
savepath = '';

% ����
min_ind = 52;   % �������0.0095-0.145Hz���ݵ�����
max_ind = 91;
number = 70;    % ��ȡn�������
Fs = 10;

% ���ڴ洢���ݵľ���
[hcN,~] = size(HC);
[mddN,~] = size(MDD);
HC_w = zeros(max_ind-min_ind+1,number,120,hcN,20);
MDD_w = zeros(max_ind-min_ind+1,number,120,mddN,20);
HC_rest = zeros(max_ind-min_ind+1,number,40,hcN,20);
MDD_rest = zeros(max_ind-min_ind+1,number,40,mddN,20);
% �洢��˳��Ϊ��ind:Ƶ�ʶΣ�number:ʱ�����У�120:����������20:8������С����8������С��ϵ����4������С�����ϵ��

% ���н������յ����ݴ���
for i = 1:hcN
    % ��ȡ������������
    lh = HC{i,1};
    lo2 = HC{i,2};
    rh = HC{i,3};
    ro2 = HC{i,4};
    syn = HC{i,5};
    
    % ��ʼ����С���任
    [~,lhw,lo2w,lWXY,lcoh] = wtc(lh,lo2,Fs);
    [~,rhw,ro2w,rWXY,rcoh] = wtc(rh,ro2,Fs);
    [~,~,~,hWXY,hcoh] = wtc(lh,rh,Fs);
    [~,~,~,o2WXY,o2coh] = wtc(lo2,ro2,Fs);
    
    % �������ݵ�����
    for j = 1:120
        num = fix(syn(2*j+1)/10);
        % ��������С����ģ
        HC_w(:,:,j,i,1) = abs(lhw(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,2) = abs(rhw(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,3) = abs(lo2w(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,4) = abs(ro2w(min_ind:max_ind,num:num+number-1));
        % �����������
        HC_w(:,:,j,i,5) = angle(lhw(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,6) = angle(rhw(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,7) = angle(lo2w(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,8) = angle(ro2w(min_ind:max_ind,num:num+number-1));
        % ����С����ģ
        HC_w(:,:,j,i,9) = abs(lWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,10) = abs(rWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,11) = abs(hWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,12) = abs(o2WXY(min_ind:max_ind,num:num+number-1));
        %����С�����
        HC_w(:,:,j,i,13) = angle(lWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,14) = angle(rWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,15) = angle(hWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,16) = angle(o2WXY(min_ind:max_ind,num:num+number-1));
        % ���ϵ��
        HC_w(:,:,j,i,17) = lcoh(min_ind:max_ind,num:num+number-1);
        HC_w(:,:,j,i,18) = rcoh(min_ind:max_ind,num:num+number-1);
        HC_w(:,:,j,i,19) = hcoh(min_ind:max_ind,num:num+number-1);
        HC_w(:,:,j,i,20) = o2coh(min_ind:max_ind,num:num+number-1);
    end
    % ��Ϣʱ�����ȡ��ʼ��Ϣ20s������
    num = fix(syn(243)/10)+200;
    
    % �������ݵ�����
    for j = 1:40
        % ��������С����ģ
        HC_rest(:,:,j,i,1) = abs(lhw(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,2) = abs(rhw(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,3) = abs(lo2w(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,4) = abs(ro2w(min_ind:max_ind,num:num+number-1));
        % �����������
        HC_rest(:,:,j,i,5) = angle(lhw(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,6) = angle(rhw(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,7) = angle(lo2w(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,8) = angle(ro2w(min_ind:max_ind,num:num+number-1));
        % ����С����ģ
        HC_rest(:,:,j,i,9) = abs(lWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,10) = abs(rWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,11) = abs(hWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,12) = abs(o2WXY(min_ind:max_ind,num:num+number-1));
        %����С�����
        HC_rest(:,:,j,i,13) = angle(lWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,14) = angle(rWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,15) = angle(hWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,16) = angle(o2WXY(min_ind:max_ind,num:num+number-1));
        % ���ϵ��
        HC_rest(:,:,j,i,17) = lcoh(min_ind:max_ind,num:num+number-1);
        HC_rest(:,:,j,i,18) = rcoh(min_ind:max_ind,num:num+number-1);
        HC_rest(:,:,j,i,19) = hcoh(min_ind:max_ind,num:num+number-1);
        HC_rest(:,:,j,i,20) = o2coh(min_ind:max_ind,num:num+number-1);
        % 50%�����ص�
        num = num + 35;
    end
end

% ��������֢����������ݴ���
for i = 1:mddN
    % ��ȡ������������
    lh = MDD{i,1};
    lo2 = MDD{i,2};
    rh = MDD{i,3};
    ro2 = MDD{i,4};
    syn = MDD{i,5};
    
    % ��ʼ����С���任
    [~,lhw,lo2w,lWXY,lcoh] = wtc(lh,lo2,Fs);
    [~,rhw,ro2w,rWXY,rcoh] = wtc(rh,ro2,Fs);
    [~,~,~,hWXY,hcoh] = wtc(lh,rh,Fs);
    [~,~,~,o2WXY,o2coh] = wtc(lo2,ro2,Fs);
    
    % �������ݵ�����
    for j = 1:120
        num = fix(syn(2*j+1)/10);
        % ��������С����ģ
        MDD_w(:,:,j,i,1) = abs(lhw(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,2) = abs(rhw(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,3) = abs(lo2w(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,4) = abs(ro2w(min_ind:max_ind,num:num+number-1));
        % �����������
        MDD_w(:,:,j,i,5) = angle(lhw(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,6) = angle(rhw(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,7) = angle(lo2w(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,8) = angle(ro2w(min_ind:max_ind,num:num+number-1));
        % ����С����ģ
        MDD_w(:,:,j,i,9) = abs(lWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,10) = abs(rWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,11) = abs(hWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,12) = abs(o2WXY(min_ind:max_ind,num:num+number-1));
        %����С�����
        MDD_w(:,:,j,i,13) = angle(lWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,14) = angle(rWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,15) = angle(hWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,16) = angle(o2WXY(min_ind:max_ind,num:num+number-1));
        % ���ϵ��
        MDD_w(:,:,j,i,17) = lcoh(min_ind:max_ind,num:num+number-1);
        MDD_w(:,:,j,i,18) = rcoh(min_ind:max_ind,num:num+number-1);
        MDD_w(:,:,j,i,19) = hcoh(min_ind:max_ind,num:num+number-1);
        MDD_w(:,:,j,i,20) = o2coh(min_ind:max_ind,num:num+number-1);
    end
    
    % ��Ϣʱ�����ȡ��ʼ��Ϣ20s������
    num = fix(syn(243)/10)+200;
    
    % �������ݵ�����
    for j = 1:40
        % ��������С����ģ
        MDD_rest(:,:,j,i,1) = abs(lhw(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,2) = abs(rhw(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,3) = abs(lo2w(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,4) = abs(ro2w(min_ind:max_ind,num:num+number-1));
        % �����������
        MDD_rest(:,:,j,i,5) = angle(lhw(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,6) = angle(rhw(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,7) = angle(lo2w(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,8) = angle(ro2w(min_ind:max_ind,num:num+number-1));
        % ����С����ģ
        MDD_rest(:,:,j,i,9) = abs(lWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,10) = abs(rWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,11) = abs(hWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,12) = abs(o2WXY(min_ind:max_ind,num:num+number-1));
        %����С�����
        MDD_rest(:,:,j,i,13) = angle(lWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,14) = angle(rWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,15) = angle(hWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,16) = angle(o2WXY(min_ind:max_ind,num:num+number-1));
        % ���ϵ��
        MDD_rest(:,:,j,i,17) = lcoh(min_ind:max_ind,num:num+number-1);
        MDD_rest(:,:,j,i,18) = rcoh(min_ind:max_ind,num:num+number-1);
        MDD_rest(:,:,j,i,19) = hcoh(min_ind:max_ind,num:num+number-1);
        MDD_rest(:,:,j,i,20) = o2coh(min_ind:max_ind,num:num+number-1);
        % 50%�����ص�
        num = num + 35;
    end
    
end

% ���ݷֶ�
HC_fear = HC_w(:,:,1:40,:,:);
HC_happy = HC_w(:,:,41:80,:,:);
HC_sad = HC_w(:,:,81:120,:,:);
MDD_fear = MDD_w(:,:,1:40,:,:);
MDD_happy = MDD_w(:,:,41:80,:,:);
MDD_sad = MDD_w(:,:,81:120,:,:);

% ��������
save([savepath 'fear.mat'],'HC_fear','MDD_fear');
save([savepath 'happy.mat'],'HC_happy','MDD_happy');
save([savepath 'sad.mat'],'HC_sad','MDD_sad');
save([savepath 'rest.mat'],'HC_rest','MDD_rest');
