data = load('data.csv');

k = 2;

num_instance = size(data,1);

center_list = unidrnd(num_instance,k,1);

last_index = -1*ones(1,num_instance);

time_max = 1000;

for t = 1: time_max

    distances = ones(k,num_instance);

    for i = 1: k
       tmp = ones(1, num_instance);
       center_matrix = tmp'*data(center_list(i),:);
       
       distances(i,:) = sqrt(sum((data - center_matrix).*(data - center_matrix), 2));
    end

    [distance_min index] = min(distances);

    if index == last_index
        break;
    else
       last_index = index; 
    end
        
end

scatter(data(find(index == 1),1),data(find(index == 1),2),'r');
hold on
scatter(data(find(index == 2),1),data(find(index == 2),2),'b');

