
%% clean the enviroment
close all;
clear;
clc;
format compact;
%% read data file

[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,class] = textread('train-white.csv' , '%f%f%f%f%f%f%f%f%f%f%f%f','delimiter',',');
[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,c] = textread('test-white.csv' , '%f%f%f%f%f%f%f%f%f%f%f%f','delimiter',',');



%[input,minI,maxI] = premnmx( [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11]')  ;
[inputn,inputps]=mapminmax( [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11]') ;

%input = [f1,f2,f3,f4,f5]'
s = length( class ) ;
s_test = length(c);
sprintf( '%d',s)

output = zeros( s , 7  ) ;
output_test = zeros(s_test,7);

for i = 1 : s 
   temp1 = class(i);
   collum = temp1 - 2 ; 
   output( i , collum  ) = 1 ;
end
for i = 1 : s_test 
   temp2 = c(i);
   collum_test = temp2 - 2 ; 
   output_test( i , collum_test  ) = 1 ;
end

%net = newff( minmax(input) , [10 10 6] , {'tansig' 'logsig' 'purelin'} , 'traingdx' ) ; 
net = newff( inputn ,output',  [5 5 5 15 30] , {'logsig' 'logsig' 'tansig' 'logsig' 'logsig'}  , 'traingd' ) ; 


net.trainparam.show = 50 ;
net.trainparam.epochs = 10000;
net.trainparam.goal = 0.0001 ;
net.trainParam.lr = 0.001 ;


net = train( net, inputn , output' ) ;


%[t1,t2,t3,t4,t5,t6,t7,t8,t9,c] = textread('test.csv' , '%f%f%f%f%f%f%f%f%f%u','headerlines',1,'delimiter',',');


%testInput = tramnmx ( [t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11]' ) ;
testInput = mapminmax ( [t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11]' ) ;


Y = sim( net , testInput ) 

%accuracy caculation%
[s1 , s2] = size( Y ) ;
hitNum = 0 ;
error_matrix = zeros(7,7);
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
figure(1)
plot(1:s2,outindex,'r.');
hold on;
plot(1:s2,c,'b*');
legend('Predict','Expect');


