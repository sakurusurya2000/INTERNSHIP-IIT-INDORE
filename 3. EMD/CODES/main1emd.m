for k = 1:9
       % Create a text file name, and read the file.
    textFileName = ['F00' num2str(k) '.txt'];
    if exist(textFileName, 'file')
        fid = fopen(textFileName, 'rt');
        textData = fread(fid);
      Z(k,:)=emdprog(textData);
        fclose(fid);
        else
        fprintf('File %s does not exist.\n', textFileName);
    end
end
    for k = 10:99
       % Create a text file name, and read the file.
    textFileName = ['F0' num2str(k) '.txt'];
    if exist(textFileName, 'file')
        fid = fopen(textFileName, 'rt');
        textData = fread(fid);
     Z(k,:)=emdprog(textData);
        fclose(fid);
    else
        fprintf('File %s does not exist.\n', textFileName);
    end
    end
    for k = 100
       % Create a text file name, and read the file.
    textFileName = ['F' num2str(k) '.txt'];
    if exist(textFileName, 'file')
        fid = fopen(textFileName, 'rt');
        textData = fread(fid);
        Z(k,:)=emdprog(textData);
        fclose(fid);
    else
        fprintf('File %s does not exist.\n', textFileName);
    end
    end    