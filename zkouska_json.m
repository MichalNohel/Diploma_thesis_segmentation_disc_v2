clc;clear all;close all;


s = struct();
s.double_array = rand(20);
s.string = 'blablabla';
s.cell = {'sdfsdf',20};

json_data = jsonencode(s);

fname = 'test.json';

fileID = fopen(fname,'w');
fprintf(fileID, json_data);
fclose(fileID);

json_res = jsondecode(fileread(fname));
