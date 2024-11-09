%% 数据规整

% 读取数据
load('fear.mat');
load('happy.mat');
load('sad.mat');
load('rest.mat');

% 创建存储矩阵
fear = zeros(40,70,40,49,20);
happy = zeros(40,70,40,49,20);
sad = zeros(40,70,40,49,20);
rest = zeros(40,70,40,49,20);
y = zeros(49,1);

% 开始储存循环
for i = 1:7
    fear(:,:,:,7*(i-1) + 1,:) = HC_fear(:,:,:,4*(i-1) + 1,:);
    fear(:,:,:,7*(i-1) + 2,:) = HC_fear(:,:,:,4*(i-1) + 2,:);
    fear(:,:,:,7*(i-1) + 3,:) = HC_fear(:,:,:,4*(i-1) + 3,:);
    fear(:,:,:,7*(i-1) + 4,:) = HC_fear(:,:,:,4*(i-1) + 4,:);
    fear(:,:,:,7*(i-1) + 5,:) = MDD_fear(:,:,:,3*(i-1) + 1,:);
    fear(:,:,:,7*(i-1) + 6,:) = MDD_fear(:,:,:,3*(i-1) + 2,:);
    fear(:,:,:,7*(i-1) + 7,:) = MDD_fear(:,:,:,3*(i-1) + 3,:);

    happy(:,:,:,7*(i-1) + 1,:) = HC_happy(:,:,:,4*(i-1) + 1,:);
    happy(:,:,:,7*(i-1) + 2,:) = HC_happy(:,:,:,4*(i-1) + 2,:);
    happy(:,:,:,7*(i-1) + 3,:) = HC_happy(:,:,:,4*(i-1) + 3,:);
    happy(:,:,:,7*(i-1) + 4,:) = HC_happy(:,:,:,4*(i-1) + 4,:);
    happy(:,:,:,7*(i-1) + 5,:) = MDD_happy(:,:,:,3*(i-1) + 1,:);
    happy(:,:,:,7*(i-1) + 6,:) = MDD_happy(:,:,:,3*(i-1) + 2,:);
    happy(:,:,:,7*(i-1) + 7,:) = MDD_happy(:,:,:,3*(i-1) + 3,:);

    sad(:,:,:,7*(i-1) + 1,:) = HC_sad(:,:,:,4*(i-1) + 1,:);
    sad(:,:,:,7*(i-1) + 2,:) = HC_sad(:,:,:,4*(i-1) + 2,:);
    sad(:,:,:,7*(i-1) + 3,:) = HC_sad(:,:,:,4*(i-1) + 3,:);
    sad(:,:,:,7*(i-1) + 4,:) = HC_sad(:,:,:,4*(i-1) + 4,:);
    sad(:,:,:,7*(i-1) + 5,:) = MDD_sad(:,:,:,3*(i-1) + 1,:);
    sad(:,:,:,7*(i-1) + 6,:) = MDD_sad(:,:,:,3*(i-1) + 2,:);
    sad(:,:,:,7*(i-1) + 7,:) = MDD_sad(:,:,:,3*(i-1) + 3,:);

    rest(:,:,:,7*(i-1) + 1,:) = HC_rest(:,:,:,4*(i-1) + 1,:);
    rest(:,:,:,7*(i-1) + 2,:) = HC_rest(:,:,:,4*(i-1) + 2,:);
    rest(:,:,:,7*(i-1) + 3,:) = HC_rest(:,:,:,4*(i-1) + 3,:);
    rest(:,:,:,7*(i-1) + 4,:) = HC_rest(:,:,:,4*(i-1) + 4,:);
    rest(:,:,:,7*(i-1) + 5,:) = MDD_rest(:,:,:,3*(i-1) + 1,:);
    rest(:,:,:,7*(i-1) + 6,:) = MDD_rest(:,:,:,3*(i-1) + 2,:);
    rest(:,:,:,7*(i-1) + 7,:) = MDD_rest(:,:,:,3*(i-1) + 3,:);

    y(7*(i-1) + 5,:) = 1;
    y(7*(i-1) + 6,:) = 1;
    y(7*(i-1) + 7,:) = 1;
end

% 保存
path = '';
save([path 'fear.mat'],'fear');
save([path 'happy.mat'],'happy');
save([path 'sad.mat'],'sad');
save([path 'rest.mat'],'rest');
save([path 'label_simple.mat'],'y');
