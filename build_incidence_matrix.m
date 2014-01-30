%function [H,W,De,Dv,f_rank] = build_incidence_matrix(file_friend,file_mentor, file_group, file_item,alpha)
function [f_rank] = build_incidence_matrix(file_friend,file_mentor, file_group, file_item,alpha)
    tt1 = now; % Current Time
    
    % Mentor Data
    M1 = load(file_mentor);
    max(M1(:,1))               
    max(M1(:,2))               
    %[dim_P ,~] = size(unique(M1(:,1)))    
    %[dim_M ,~] = size(unique(M1(:,2)))    
    
    % Friend Data
    M2 = load(file_friend);
    max(M2(:,1))               
    max(M2(:,2))                          
    %[dim_P1 ,~] = size(unique(M2(:,1)))    
    %[dim_P2 ,~] = size(unique(M2(:,2)))    
    
    % Group Data
    M3 = load(file_group);
    %max(M3(:,1))               
    %max(M3(:,2))                          
    %[dim_G ,~] = size(unique(M3(:,1)))    
    %[dim_P ,~] = size(unique(M3(:,2)))    
    
    % Item Data
    M4 = load(file_item);
    %max(M4(:,1))               
    %max(M4(:,2))                          
    %[dim_P ,~] = size(unique(M4(:,1)))    
    %[dim_I ,~] = size(unique(M4(:,2)));
    
    %Size of the Vertices set
    %P_Tot = unique([unique(M1(:,1))'  unique(M1(:,2))' unique(M2(:,1))' unique(M2(:,2))' unique(M1(:,2))' unique(M3(:,2))' unique(M4(:,1))' unique(M4(:,2))']);
    P_Tot = unique([unique(M1)'  unique(M2)' unique(M3(:,2))' unique(M4)']);
    %V_size = size(P_Tot,2)
      
    %H_row = zeros(size(P_Tot,1)+dim_I,1);
    incidence_length = 2*(size(M1,1) + size(M2,1) + size(M4,1)) + size(M3,1);
    S_h = zeros(1,incidence_length );
    I_h = zeros(1,incidence_length );
    J_h = zeros(1,incidence_length );
    %S_w = zeros(1,size(M1,1) + size(M2,1) + size(M4,1) + size(M3,1));
    
    i=1;row=1;count=1;
    while(true)
        if(M1(i,1)~=M1(i,2))
            S_h(count)=1;
            I_h(count)=row;
            J_h(count)=find(P_Tot==M1(i,1),1);
            count=count+1;
            S_h(count)=1;
            I_h(count)=row;
            J_h(count)=find(P_Tot==M1(i,2),1);
            count=count+1;
            row=row+1;
        end                         
        i=i+1;
        if(i>size(M1,1))
            break;
        end
    end
    i=1;    
    while(true)
        if(M2(i,1)~=M2(i,2))
            S_h(count)=1;
            I_h(count)=row;
            J_h(count)=find(P_Tot==M2(i,1),1);
            count=count+1;
            S_h(count)=1;
            I_h(count)=row;
            J_h(count)=find(P_Tot==M2(i,2),1);
            count=count+1;
            row=row+1;
        end                         
        i=i+1;
        if(i>size(M2,1))
            break;
        end
    end
    i=1;    
    while(true)
        if(M4(i,1)~=M4(i,2))
            S_h(count)=1;
            I_h(count)=row;
            J_h(count)=find(P_Tot==M4(i,1),1);
            count=count+1;
            S_h(count)=1;
            I_h(count)=row;
            J_h(count)=find(P_Tot==M4(i,2),1);
            count=count+1;
            row=row+1;
        end                         
        i=i+1;
        if(i>size(M4,1))
            break;
        end
    end
    i=1;
    t=M3(i,1);
    m3_size = size(M3,1);
    while(true)
        S_h(count)=1;
        I_h(count)=row;
        J_h(count)=find(P_Tot==M3(i,2),1);
        count=count+1;                    
        i=i+1;        
        if(i>m3_size)
            break;
        end        
        if(t~=M3(i,1))                               
            row=row+1;
        end        
        t=M3(i,1);                        
    end   
    %}
    display('Data Loading done.');
    
    H = sparse(I_h,J_h,S_h);
    size(H)
    n_v = size(H,2);
    n_e = size(H,1);
    row
    W = sparse(1:n_e,1:n_e,ones(1,n_e));    
    %Dv = sparse(diag(H*ones(size(H,2),1)));
    De = sparse(1:n_e,1:n_e,(H*ones(n_v,1))');
    %De = sparse(diag(H'*ones(size(H,1),1))); 
    Dv = sparse(1:n_v,1:n_v,(H'*ones(n_e,1))');
    %Y = H(:,size(unique(M1'),1)+1);
    Y = zeros(n_v,1);
    Y(size(unique(M1'),1)+1,1)=1;
    sparse_eye = sparse(1:n_v,1:n_v,ones(1,n_v));
    save('sparse_eye','sparse_eye');
    clear sparse_eye;    
    De_inverse = inv(De);
    clear De;
    save('De_inverse','De_inverse');
    clear De_inverse;
    Dv_inverse_root = sqrt(inv(Dv));
    clear Dv;
    save('W','W');
    clear W;
    save('Y','Y');
    clear Y;    
    display('Basic matrices built.');
    
    t1 = (Dv_inverse_root)* H';
    save('t1','t1');
    display('Size t1');
    size(t1)
    clear t1;
    display('t1 done.');
        
    t2 = H*(Dv_inverse_root);
    save('t2','t2');
    display('Size t2');
    size(t2)
    clear t2;
    display('t2 done.');
        
    load('W');
    load('De_inverse');
    t3 = W*De_inverse;
    display('t3 done.');
    display('Size t3');
    size(t3)
    
    load('t1');
    t4 = t1*t3;
    clear t3;
    clear t1;
    display('t4 done.');
    display('Size t4');
    size(t4)
    
    load('t2');
    t5 = t4 * t2;
    clear t4; clear t2;
    display('Size t5');
    size(t5)
    display('t5 done.');
        
    load('sparse_eye')
    t6 = sparse_eye - alpha*t5;
    clear sparse_eye; clear t5;
    display('t6 done.');
    display('Size t6');
    size(t6)
    
    %t7 = inv(t6);
    %clear t6;
    %display('t7 done.');
    
    load('Y'); 
    display('Y Loaded.');
    display('Size Y');
    size(Y)
    %f_rank = t6\Y;
    %display('All done.');
    %f_rank = inv(sparse_eye - alpha*(Dv_inverse_root)* H' * W * De_inverse * H *(Dv_inverse_root)) * Y;
    
    %{
    S_De = zeros(1,size(H,1));
    S_Dv = zeros(1,size(H,2));
    size_e=size(H,1);
    size_v=size(H,2);
    for i =1:size_e
        S_De(1,i)=size(find(H(i,:)>0),2);
    end
    for i =1:size_v
        S_Dv(1,i)=size(find(H(:,i)>0),2);
    end
    De = sparse(1:size_e,1:size_e,S_De);
    Dv = sparse(1:size_v,1:size_v,S_Dv);
    %}
                
    disp('Time taken ......');
    disp(datevec(now - tt1));    
end
