class FGM:
    def __init__(self,model):
        self.model=model
        self.backup={}
        
    def attack(self,epsilon=1,emb_name='emb'):
        for name,param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name]=param.data.clone()
                norm=torch.norm(param.grad)
                if norm!=0 and not torch.isnan(norm):
                    r_at=epsilon*param.grad/norm
                    param.data.add_(r_at)

    def restore(self,emb_name='emb'):
        for name,param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data=self.backup[name]
        self.backup={}


fgm=FGM(model)
for batch_input,batch_label in data:
    loss=model(batch_input,batch_label)
    loss.backward()#反向传播，计算梯度
    fgm.attack()#在embedding上添加扰动
    loss_adv=model(batch_input,batch_label)#计算对抗样本的loss
    loss_adv.backward()
    fgm.restore()#恢复原来的embedding
    optimizer.step()#由于之前loss的梯度并没有更新，所以这里应该是使用grad_loss+grad_loss_adv进行更新，但是两者重复计算了参数，所以
    #我觉得应该要么只用loss_adv的梯度更新，要么用两者的平均
    model.zero_grad()





class PGD:
    def __init__(self,model,emb_name,epsilon=1,alpha=0.3):
        self.model=model
        self.emb_name=emb_name
        self.epsilon=epsilon
        self.alpha=alpha
        self.emb_backup={}
        self.grad_backup={}
        
    def attack(self,is_first_attack=False):
        for name,param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name]=param.data.clone()
                norm=torch.norm(param.grad)
                if norm!=0:
                    r_at=self.alpha*param.grad/norm
                    param.data.add_(r_at)
                    param.data=self.preject(name,param.data,self.epsilon)

    def restore(self):
        for name,param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data=self.emb_backup[name]
            self.emb_backup={}
            
    def project(self,param_name,param.data,epsilon):
        r=param_data-self.emb_backup[param_name]
        if torch.norm(r)>epsilon:
            r=epsilon*r/torch.norm(r)
        return self.emb_backup[param_name]+r
        
    def back_grad(self):
        for name,param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name]=param.grad.clone()

    def restore_grad(self):
        for name,param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad=self.grad_backup[name]





pgd=PGD(model)
k=3
for batch_input,batch_label in data:
    loss=model(batch_input,batch_label)
    loss.backward()
    pgd.backup_grad()#备份初始grad
    for t in range(k):
        pgd.attack(is_first_attack=(t==0))
        if t!=k-1:
            model.zero_grad()#每一步都将模型的梯度清零，此时原始梯度也没了，即使用添加扰动的参数计算梯度，也就是该步的梯度不会受到前一步扰动的影响
            #相当于扰动也在不断的被优化--使loss更大
        else:
            pgd.restore_grad()#还原初始grad，实现原始梯度与最后一步扰动的梯度相加
        loss_adv=model(batch_input,batch_label)
        loss_adv.backward()
    pgd.restore()#还原初始embedding
    optimizer.step()#用原始梯度+最后一步加扰动的梯度来更新参数
    model.zero_grad()

