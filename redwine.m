
%% clean the enviroment
close all;
clear;
clc;
format compact;
%% Read the data and load the data
[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,class] = textread('train-red.csv' , '%f%f%f%f%f%f%f%f%f%f%f%f','delimiter',',');%load training data and set first 11 colunms as input and the last one is class label
[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,c] = textread('test-red.csv' , '%f%f%f%f%f%f%f%f%f%f%f%f','delimiter',',');%load test data and set first 11 colunms as input and the last one is class label

[inputn,inputps]=mapminmax( [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11]') ;%do the preprocess:processes matrices by normalizing the minimum and maximum values
testInput = mapminmax ( [t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11]' ) ;%normalizing the test input
%% Generate the output matrices
s = length( class ) ;%the number of the colunms should be the same as class number
s_test = length(c);% A test output, help to generate the error matrix
%sprintf( '%d',s)

output = zeros( s , 6  ) ;%initialize the output matrix
output_test = zeros(s_test,6);%initialize the test output matrix

%% initialization: use position to prsent the lable
for i = 1 : s 
   temp = class(i);
   collum = temp - 2 ; 
   output( i , collum  ) = 1 ;
end
for i = 1 : s_test 
   temp = c(i);
   collum_test = temp - 2 ; 
   output_test( i , collum_test  ) = 1 ;
end

%% set to generate the network 
net = newff( inputn ,output',  [5 5 15 30] , {'logsig' 'tansig' 'logsig' 'logsig'}  , 'traingd' ) ; 

%% network setting
net.trainparam.show = 50 ;%the show gap
net.trainparam.epochs = 5000;% epochs used
net.trainparam.goal = 0.0001 ;%the error goals
net.trainParam.lr = 0.001 ;%the learning rate

net = train( net, inputn , output' ) ;%start training

%% start simulation
Y = sim( net , testInput ) 

%% accuracy caculation%
[s1 , s2] = size( Y ) ;
hitNum = 0 ;
error_matrix = zeros(6,6);
outindex = zeros(1,s2);
for i = 1 : s2
    [m , Index] = max( Y( : ,  i ) ) ;
    Index = Index + 2
    outindex(1,i) = Index
    if( Index  == c(i)   ) 
        hitNum = hitNum + 1 ;
        label_index = c(i) - 2 ;
        error_matrix(label_index,label_index) = error_matrix(label_index,label_index) + 1;
    else
        Index = Index - 2;
        label_index = c(i) - 2 ;
        error_matrix(label_index,Index) = error_matrix(label_index,Index) + 1;
    end
end
sprintf('The accuracy is %3.3f%%',100 * hitNum / s2 )


%The distribution for prediction and expection%
figure(1)%draw a plot
plot(1:s2,outindex,'r.');%first is predict
hold on;
plot(1:s2,c,'b*');
legend('Predict','Expect');


