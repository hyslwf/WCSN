function varargout=wtc(x,y,Fs)
%% 修改版小波相干计算，输出单条曲线小波系数、交叉小波系数和小波相干谱

% ------validate and reformat timeseries.
dt = 1/Fs;
n=length(x);

%----------default arguments for the wavelet transform-----------
Dj=1/12;

%-----------:::::::::::::--------- ANALYZE ----------::::::::::::------------
[X,f,~] = cwt(x,'amor',Fs);
[Y,~,~] = cwt(y,'amor',Fs);

scale=Fs*centfrq('morl')./f;
sinv=1./(scale');

%Smooth X and Y before truncating!  (minimize coi)
sX=smoothwavelet(sinv(:,ones(1,n)).*(abs(X).^2),dt,Dj,scale);
sY=smoothwavelet(sinv(:,ones(1,n)).*(abs(Y).^2),dt,Dj,scale);

% -------- Cross wavelet -------
Wxy=X.*conj(Y);

% ----------------------- Wavelet coherence ---------------------------------
sWxy=smoothwavelet(sinv(:,ones(1,n)).*Wxy,dt,Dj,scale);
Coh=abs(sWxy).^2./(sX.*sY);


varargout={f,X,Y,Wxy,Coh};
% varargout={X,Y,f,period,scale,coi,sinv,Wxy,Coh};
varargout=varargout(1:nargout);