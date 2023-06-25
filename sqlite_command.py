import sqlite3


def insert_pretrain(config):
    # 连接数据库
    conn = sqlite3.connect('SimCLR_result.db')
    cur = conn.cursor()

    # 创建表
    create_table = "create table if not exists SimCLR_result(no INTEGER PRIMARY KEY AUTOINCREMENT, type text, " \
                   "path text, pretrain_path text, dataset text, train_num number, model text, add_dense int, " \
                   "batch_size number, epoch number, lr number, out_dim number, shift number, cut number, " \
                   "filter number, GE number, GE_epoch number, model_eval int, pretrain_no int, frozen int, " \
                   "pretrain_train_num int)"
    cur.execute(create_table)

    # 插入数据
    data = (None, 'pretrain', config['outfile'], None, config['common']['dataset_name'], None,
            config['common']['model_name'], 0, config['common']['batch_size'], config['epoch'], config['lr'],
            config['out_dim'], config["augmentation"]['data_shift'], config["augmentation"]['data_cut'],
            config["augmentation"]['data_filter'], None, None, None, None, None, config["train_num"])
    order = "INSERT INTO SimCLR_result VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"
    conn.execute(order, data)
    conn.commit()

    # 获取no
    select_no = "SELECT no FROM SimCLR_result WHERE path = '" + config['outfile'] + "'"
    cur.execute(select_no)
    pretrain_no = cur.fetchall()[0][0]  # 获取查询结果一般可用.fetchone()方法（获取第一条），或者用.fetchall()方法（获取所有条）

    # 关闭连接
    cur.close()
    conn.close()

    return pretrain_no


def insert_tuning(config, best_GE, best_GE_epoch):
    # 连接数据库
    conn = sqlite3.connect('SimCLR_result.db')
    cur = conn.cursor()

    if config['pretrain_path'] is None:
        pretrain_no = None
    else:
        select_no = "SELECT no FROM SimCLR_result WHERE path = '" + config['pretrain_path'] + "'"
        cur.execute(select_no)
        pretrain_no = cur.fetchall()[0][0]

    # 插入数据
    data = (None, 'tuning', config['outfile'], config['pretrain_path'], config['common']['dataset_name'],
            config['train_num'], config['common']['model_name'], 1 if config['add_dense_bool'] else 0,
            config['common']['batch_size'], config['epoch'], config['lr'], config['out_dim'], None, None, None, best_GE,
            best_GE_epoch, 1 if config['model_eval'] else 0, pretrain_no, 1 if config['frozen'] else 0, None)
    order = "INSERT INTO SimCLR_result VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"
    conn.execute(order, data)
    conn.commit()

    # 获取no
    select_no = "SELECT no FROM SimCLR_result WHERE path = '" + config['outfile'] + "'"
    cur.execute(select_no)
    tuning_no = cur.fetchall()[0][0]  # 获取查询结果一般可用.fetchone()方法（获取第一条），或者用.fetchall()方法（获取所有条）

    # 关闭连接
    cur.close()
    conn.close()

    return tuning_no


def insert_network(config, best_GE, best_GE_epoch):
    # 连接数据库
    conn = sqlite3.connect('SimCLR_result.db')
    cur = conn.cursor()

    # 创建表
    create_table = "create table if not exists SimCLR_result(no INTEGER PRIMARY KEY AUTOINCREMENT, type text, " \
                   "path text, pretrain_path text, dataset text, train_num number, model text, add_dense int, " \
                   "batch_size number, epoch number, lr number, out_dim number, shift number, cut number, " \
                   "filter number, GE number, GE_epoch number, model_eval int, pretrain_no int, frozen int, " \
                   "pretrain_train_num int)"
    cur.execute(create_table)

    # 插入数据
    data = (None, 'network', config['outfile'], None, config['common']['dataset_name'], config['train_num'],
            config['common']['model_name'], 1 if config['add_dense_bool'] else 0, config['common']['batch_size'],
            config['epoch'], config['lr'], config['out_dim'], None, None, None, best_GE, best_GE_epoch,
            1 if config['model_eval'] else 0, None, 1 if config['frozen'] else 0, None)
    order = "INSERT INTO SimCLR_result VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"
    conn.execute(order, data)
    conn.commit()

    # 获取no
    select_no = "SELECT no FROM SimCLR_result WHERE path = '" + config['outfile'] + "'"
    cur.execute(select_no)
    network_no = cur.fetchall()[0][0]  # 获取查询结果一般可用.fetchone()方法（获取第一条），或者用.fetchall()方法（获取所有条）

    # 关闭连接
    cur.close()
    conn.close()

    return network_no


def select_path_from_no(no):
    # 连接表
    conn = sqlite3.connect('SimCLR_result.db')
    cur = conn.cursor()

    # 寻找no对应路径
    select_path = "SELECT path FROM SimCLR_result WHERE no = " + str(no)
    cur.execute(select_path)
    path = cur.fetchall()[0][0]  # 获取查询结果一般可用.fetchone()方法（获取第一条），或者用.fetchall()方法（获取所有条）

    # 关闭连接
    cur.close()
    conn.close()

    return path

def select_tuning_from_pretrain(pretrain_no):
    # 连接表
    conn = sqlite3.connect('SimCLR_result.db')
    cur = conn.cursor()

    # 寻找no对应路径
    select_tuning = "SELECT no FROM SimCLR_result WHERE pretrain_no = " + str(pretrain_no)
    cur.execute(select_tuning)
    result = cur.fetchall()
    tuning = [i[0] for i in result]  # 获取查询结果一般可用.fetchone()方法（获取第一条），或者用.fetchall()方法（获取所有条）

    # 关闭连接
    cur.close()
    conn.close()

    return tuning


def change_GE(no, GE, GE_epoch):
    # 连接表
    conn = sqlite3.connect('SimCLR_result.db')
    cur = conn.cursor()

    update_GE = "UPDATE SimCLR_result SET GE = " + str(GE) + ", GE_epoch = " + str(GE_epoch) + " where no = " + str(no)
    cur.execute(update_GE)
    conn.commit()

    # 关闭连接
    cur.close()
    conn.close()
