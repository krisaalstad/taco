function [pp] = percplot(x,Y,c,tr,lst,ec)
% percplot(x,Y,c,tr,lst);
if size(x,2)==1
    x=x';
end
if size(Y,2)==3
    Y=Y(:,1:2:3);
elseif size(Y,1)==3
    Y=Y(1:2:3,:);
end
if size(Y,1)~=2
    Y=Y';
    if size(Y,1)~=2
        error('Expected Y to contain a percentile range: size(Y)=2(3)xN or Nx2(3)');
    end
end
 
xf=[x fliplr(x)];
Yf=[Y(2,:) fliplr(Y(1,:))];
%error('stop');
%xf=[x(1:(end-1)); x(2:end); x(2:end); x(1:(end-1))];
%Yf = [Y(2,1:(end-1)); Y(2,2:end); Y(1,2:end); Y(1,1:(end-1))];



% Handle NaNs better




%pp=fill(xf,Yf,c,'LineStyle',lst,'EdgeColor',ec);

%c(end)=1;
if sum(c)==3
    lw=0.5;
else
    lw=1;
end
pp=fill(xf,Yf,c,'LineStyle',lst,'EdgeColor',ec,'LineWidth',lw);
%tr=1;
alpha(tr);