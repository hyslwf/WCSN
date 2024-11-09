%% 对滤波数据进行小波变换和数据截断

% 读取滤波数据
load('data.mat');

% 存储路径
savepath = '';

% 参数
min_ind = 52;   % 经计算得0.0095-0.145Hz数据的索引
max_ind = 91;
number = 70;    % 截取n秒的数据
Fs = 10;

% 用于存储数据的矩阵
[hcN,~] = size(HC);
[mddN,~] = size(MDD);
HC_w = zeros(max_ind-min_ind+1,number,120,hcN,20);
MDD_w = zeros(max_ind-min_ind+1,number,120,mddN,20);
HC_rest = zeros(max_ind-min_ind+1,number,40,hcN,20);
MDD_rest = zeros(max_ind-min_ind+1,number,40,mddN,20);
% 存储的顺序为：ind:频率段，number:时间序列，120:样本个数，20:8个单独小波，8个交叉小波系数，4个交叉小波相干系数

% 进行健康对照的数据处理
for i = 1:hcN
    % 读取单个被试数据
    lh = HC{i,1};
    lo2 = HC{i,2};
    rh = HC{i,3};
    ro2 = HC{i,4};
    syn = HC{i,5};
    
    % 开始进行小波变换
    [~,lhw,lo2w,lWXY,lcoh] = wtc(lh,lo2,Fs);
    [~,rhw,ro2w,rWXY,rcoh] = wtc(rh,ro2,Fs);
    [~,~,~,hWXY,hcoh] = wtc(lh,rh,Fs);
    [~,~,~,o2WXY,o2coh] = wtc(lo2,ro2,Fs);
    
    % 保存数据到矩阵
    for j = 1:120
        num = fix(syn(2*j+1)/10);
        % 单个曲线小波的模
        HC_w(:,:,j,i,1) = abs(lhw(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,2) = abs(rhw(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,3) = abs(lo2w(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,4) = abs(ro2w(min_ind:max_ind,num:num+number-1));
        % 单个曲线相角
        HC_w(:,:,j,i,5) = angle(lhw(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,6) = angle(rhw(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,7) = angle(lo2w(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,8) = angle(ro2w(min_ind:max_ind,num:num+number-1));
        % 交叉小波的模
        HC_w(:,:,j,i,9) = abs(lWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,10) = abs(rWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,11) = abs(hWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,12) = abs(o2WXY(min_ind:max_ind,num:num+number-1));
        %交叉小波相角
        HC_w(:,:,j,i,13) = angle(lWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,14) = angle(rWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,15) = angle(hWXY(min_ind:max_ind,num:num+number-1));
        HC_w(:,:,j,i,16) = angle(o2WXY(min_ind:max_ind,num:num+number-1));
        % 相关系数
        HC_w(:,:,j,i,17) = lcoh(min_ind:max_ind,num:num+number-1);
        HC_w(:,:,j,i,18) = rcoh(min_ind:max_ind,num:num+number-1);
        HC_w(:,:,j,i,19) = hcoh(min_ind:max_ind,num:num+number-1);
        HC_w(:,:,j,i,20) = o2coh(min_ind:max_ind,num:num+number-1);
    end
    % 休息时间戳，取开始休息20s后数据
    num = fix(syn(243)/10)+200;
    
    % 保存数据到矩阵
    for j = 1:40
        % 单个曲线小波的模
        HC_rest(:,:,j,i,1) = abs(lhw(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,2) = abs(rhw(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,3) = abs(lo2w(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,4) = abs(ro2w(min_ind:max_ind,num:num+number-1));
        % 单个曲线相角
        HC_rest(:,:,j,i,5) = angle(lhw(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,6) = angle(rhw(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,7) = angle(lo2w(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,8) = angle(ro2w(min_ind:max_ind,num:num+number-1));
        % 交叉小波的模
        HC_rest(:,:,j,i,9) = abs(lWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,10) = abs(rWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,11) = abs(hWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,12) = abs(o2WXY(min_ind:max_ind,num:num+number-1));
        %交叉小波相角
        HC_rest(:,:,j,i,13) = angle(lWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,14) = angle(rWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,15) = angle(hWXY(min_ind:max_ind,num:num+number-1));
        HC_rest(:,:,j,i,16) = angle(o2WXY(min_ind:max_ind,num:num+number-1));
        % 相关系数
        HC_rest(:,:,j,i,17) = lcoh(min_ind:max_ind,num:num+number-1);
        HC_rest(:,:,j,i,18) = rcoh(min_ind:max_ind,num:num+number-1);
        HC_rest(:,:,j,i,19) = hcoh(min_ind:max_ind,num:num+number-1);
        HC_rest(:,:,j,i,20) = o2coh(min_ind:max_ind,num:num+number-1);
        % 50%窗口重叠
        num = num + 35;
    end
end

% 进行抑郁症患者组的数据处理
for i = 1:mddN
    % 读取单个被试数据
    lh = MDD{i,1};
    lo2 = MDD{i,2};
    rh = MDD{i,3};
    ro2 = MDD{i,4};
    syn = MDD{i,5};
    
    % 开始进行小波变换
    [~,lhw,lo2w,lWXY,lcoh] = wtc(lh,lo2,Fs);
    [~,rhw,ro2w,rWXY,rcoh] = wtc(rh,ro2,Fs);
    [~,~,~,hWXY,hcoh] = wtc(lh,rh,Fs);
    [~,~,~,o2WXY,o2coh] = wtc(lo2,ro2,Fs);
    
    % 保存数据到矩阵
    for j = 1:120
        num = fix(syn(2*j+1)/10);
        % 单个曲线小波的模
        MDD_w(:,:,j,i,1) = abs(lhw(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,2) = abs(rhw(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,3) = abs(lo2w(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,4) = abs(ro2w(min_ind:max_ind,num:num+number-1));
        % 单个曲线相角
        MDD_w(:,:,j,i,5) = angle(lhw(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,6) = angle(rhw(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,7) = angle(lo2w(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,8) = angle(ro2w(min_ind:max_ind,num:num+number-1));
        % 交叉小波的模
        MDD_w(:,:,j,i,9) = abs(lWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,10) = abs(rWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,11) = abs(hWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,12) = abs(o2WXY(min_ind:max_ind,num:num+number-1));
        %交叉小波相角
        MDD_w(:,:,j,i,13) = angle(lWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,14) = angle(rWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,15) = angle(hWXY(min_ind:max_ind,num:num+number-1));
        MDD_w(:,:,j,i,16) = angle(o2WXY(min_ind:max_ind,num:num+number-1));
        % 相关系数
        MDD_w(:,:,j,i,17) = lcoh(min_ind:max_ind,num:num+number-1);
        MDD_w(:,:,j,i,18) = rcoh(min_ind:max_ind,num:num+number-1);
        MDD_w(:,:,j,i,19) = hcoh(min_ind:max_ind,num:num+number-1);
        MDD_w(:,:,j,i,20) = o2coh(min_ind:max_ind,num:num+number-1);
    end
    
    % 休息时间戳，取开始休息20s后数据
    num = fix(syn(243)/10)+200;
    
    % 保存数据到矩阵
    for j = 1:40
        % 单个曲线小波的模
        MDD_rest(:,:,j,i,1) = abs(lhw(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,2) = abs(rhw(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,3) = abs(lo2w(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,4) = abs(ro2w(min_ind:max_ind,num:num+number-1));
        % 单个曲线相角
        MDD_rest(:,:,j,i,5) = angle(lhw(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,6) = angle(rhw(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,7) = angle(lo2w(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,8) = angle(ro2w(min_ind:max_ind,num:num+number-1));
        % 交叉小波的模
        MDD_rest(:,:,j,i,9) = abs(lWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,10) = abs(rWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,11) = abs(hWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,12) = abs(o2WXY(min_ind:max_ind,num:num+number-1));
        %交叉小波相角
        MDD_rest(:,:,j,i,13) = angle(lWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,14) = angle(rWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,15) = angle(hWXY(min_ind:max_ind,num:num+number-1));
        MDD_rest(:,:,j,i,16) = angle(o2WXY(min_ind:max_ind,num:num+number-1));
        % 相关系数
        MDD_rest(:,:,j,i,17) = lcoh(min_ind:max_ind,num:num+number-1);
        MDD_rest(:,:,j,i,18) = rcoh(min_ind:max_ind,num:num+number-1);
        MDD_rest(:,:,j,i,19) = hcoh(min_ind:max_ind,num:num+number-1);
        MDD_rest(:,:,j,i,20) = o2coh(min_ind:max_ind,num:num+number-1);
        % 50%窗口重叠
        num = num + 35;
    end
    
end

% 数据分段
HC_fear = HC_w(:,:,1:40,:,:);
HC_happy = HC_w(:,:,41:80,:,:);
HC_sad = HC_w(:,:,81:120,:,:);
MDD_fear = MDD_w(:,:,1:40,:,:);
MDD_happy = MDD_w(:,:,41:80,:,:);
MDD_sad = MDD_w(:,:,81:120,:,:);

% 保存数据
save([savepath 'fear.mat'],'HC_fear','MDD_fear');
save([savepath 'happy.mat'],'HC_happy','MDD_happy');
save([savepath 'sad.mat'],'HC_sad','MDD_sad');
save([savepath 'rest.mat'],'HC_rest','MDD_rest');
