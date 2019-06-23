for k = 1:9
       % Create a text file name, and read the file.
    textFileName = ['O00' num2str(k) '.txt'];
    if exist(textFileName, 'file')
        fid = fopen(textFileName, 'rt');
        textData = fread(fid);
      Z(k,:)=TWT(textData);
        fclose(fid);
        else
        fprintf('File %s does not exist.\n', textFileName);
    end
end
    for k = 10:99
       % Create a text file name, and read the file.
    textFileName = ['O0' num2str(k) '.txt'];
    if exist(textFileName, 'file')
        fid = fopen(textFileName, 'rt');
        textData = fread(fid);
      Z(k,:)=TWT(textData);
        fclose(fid);
    else
        fprintf('File %s does not exist.\n', textFileName);
    end
    end
    for k = 100
       % Create a text file name, and read the file.
    textFileName = ['O' num2str(k) '.txt'];
    if exist(textFileName, 'file')
        fid = fopen(textFileName, 'rt');
        textData = fread(fid);
        Z(k,:)=TWT(textData);
        fclose(fid);
    else
        fprintf('File %s does not exist.\n', textFileName);
    end
    end