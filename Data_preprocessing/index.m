%% 获取截断数据索引

% 读取滤波数据
load('data.mat','HC');

% 记录索引
min_ind = 0;
max_ind = 0;

% 计算频率f
lh = HC{11,1};
lo2 = HC{11,2};

[f,x,~,~,~] = wtc(lh,lo2,10);

% 0.0095–0.021, 0.021–0.052, 0.052–0.145, 0.145–0.6, and 0.6–1.6 Hz，选前三个频率
T1 = 1;T2 = 1;
for i = 1:length(f)
    if (f(i) <= 0.145 && T1 == 1)
        min_ind = i;T1 = 0;
    end
    if (f(i) <= 0.0095 && T2 == 1)
        max_ind = i;T2 = 0;
    end
end
save('index.mat','max_ind','min_ind');
    


