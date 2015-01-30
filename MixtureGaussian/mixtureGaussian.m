data = load('data.csv');
k =2;

threshold = 0.00000001;
[num_instance, num_dim] = size(data);

mu = data(1:2,:);
weight = [1 1];
sigma = zeros(num_dim, num_dim, k);
sigma(:,:,1) = [1 0; 0 1];
sigma(:,:,2) = [1 0; 0 1];

last_mu = -1000000;
count = 0;
while 1
    count = count + 1
    px = zeros(num_instance, k);
    for i = 1:k
        
        for j = 1:num_instance
            diff = data(j,:) - mu(i, :);
            inv_pSigma = inv(sigma(:, :, i));
            tmp = sum((diff*inv_pSigma) .* diff, 2);
            coef = (2*pi)^(-num_dim/2) * sqrt(det(inv_pSigma));
            px(j, i) = coef * exp(-0.5*tmp);
            
        end
    end
    
    
    p = px .* repmat(weight, num_instance, 1);
    p = p ./ repmat(sum(p, 2), 1, k);
    
    
    P = sum(p, 1);
    
    for i = 1:k
        tmp_mu = zeros(1,2);
        for j = 1: num_instance
            tmp_mu = tmp_mu + p(j, i)*data(j,:)/P(i);
        end
        mu(:,i) = tmp_mu;
    end
    
    weight = P/num_instance;
    for i = 1:k
        tmp_sigma = zeros(2, 2);
        for j = 1: num_instance
            diff = data(j,:)-mu(i, :);
            tmp_sigma = tmp_sigma + (diff' * p(j,i) * diff) / P(i);
        end
        sigma(:, :, i) = tmp_sigma;
    end
    
    
    gap = abs(max(mu - last_mu));
    if gap < threshold
        break
    end
    last_mu = mu;
end




num_instance = size(data,1);

index = ones(1,num_instance);
index(find(px(:,1)>px(:,2))) = 1;
index(find(px(:,1)<px(:,2))) = 2;

scatter(data(find(index == 1),1),data(find(index == 1),2),'r');

hold on

a=sigma(1,1,1);
b=sigma(2,2,1);
x0=mu(1,1); % x0,y0 ellipse centre coordinates
y0=mu(1,2);
t=-pi:0.01:pi;
x=x0+a*cos(t);
y=y0+b*sin(t);
plot(x,y)

hold on
scatter(data(find(index == 2),1),data(find(index == 2),2),'b');

hold on
a=sigma(1,1,2);
b=sigma(2,2,2);
x0=mu(2,1); % x0,y0 ellipse centre coordinates
y0=mu(2,2);
t=-pi:0.01:pi;
x=x0+a*cos(t);
y=y0+b*sin(t);
plot(x,y)

hold on
plot(mu(1,:),'*')
hold on
plot(mu(2,:),'*')



