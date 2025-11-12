import sqlite3


def insert_pretrain(config):
    # 连接数据库
    conn = sqlite3.connect('SimCLR_result.db')
    cur = conn.cursor()
    cur.execute('PRAGMA busy_timeout = 10000')

    # 创建表
    create_table = "create table if not exists SimCLR_result(no INTEGER PRIMARY KEY AUTOINCREMENT, type text, " \
                   "path text, pretrain_path text, dataset text, train_num number, model text, add_dense int, " \
                   "batch_size number, epoch number, lr number, out_dim number, shift number, cut number, " \
                   "filter number, GE number, GE_epoch number, model_eval int, pretrain_no int, frozen int, " \
                   "pretrain_train_num int, tuning_optim text, projection_layer int)"
    cur.execute(create_table)

    # 插入数据
    data = (None, 'pretrain', config['outfile'], None, config['common']['dataset_name'], None,
            config['common']['model_name'], 0, config['common']['batch_size'], config['epoch'], config['lr'],
            config['out_dim'], 'data_shift' in config["augmentation"] and config["augmentation"]['data_shift'],
            'data_cut' in config["augmentation"] and config["augmentation"]['data_cut'],
            'data_filter' in config["augmentation"] and config["augmentation"]['data_filter'], None, None, None, None,
            None, config["train_num"], None, config["projection_head_layer"])
    order = "INSERT INTO SimCLR_result VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"
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


def insert_network(config, best_GE, best_GE_epoch):
    # 连接数据库
    conn = sqlite3.connect('SimCLR_result.db')
    cur = conn.cursor()
    cur.execute('PRAGMA busy_timeout = 10000')

    # 创建表
    create_table = "create table if not exists SimCLR_result(no INTEGER PRIMARY KEY AUTOINCREMENT, type text, " \
                   "path text, pretrain_path text, dataset text, train_num number, model text, add_dense int, " \
                   "batch_size number, epoch number, lr number, out_dim number, shift number, cut number, " \
                   "filter number, GE number, GE_epoch number, model_eval int, pretrain_no int, frozen int, " \
                   "pretrain_train_num int, tuning_optim text, projection_layer int)"
    cur.execute(create_table)

    # 插入数据
    if config['pretrain_path'] is None or config['pretrain_path'] == '':
        pretrain_no = None
        config['pretrain_path'] = None
    else:
        select_no = "SELECT no FROM SimCLR_result WHERE path = '" + config['pretrain_path'] + "'"
        cur.execute(select_no)
        pretrain_no = cur.fetchall()[0][0]

    # 插入数据
    data = (None, config['common']['experiment_type'], config['outfile'], config['pretrain_path'],
            config['common']['dataset_name'], config['train_num'], config['common']['model_name'],
            1 if config['add_dense_bool'] else 0, config['common']['batch_size'], config['epoch'], config['lr'],
            config['out_dim'], None, None, None, best_GE, best_GE_epoch, 1 if config['model_eval'] else 0,
            pretrain_no, 1 if config['frozen'] else 0, None, config['optim'], None)
    order = "INSERT INTO SimCLR_result VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"
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
    cur.execute('PRAGMA busy_timeout = 10000')

    # 寻找no对应路径
    select_path = "SELECT path FROM SimCLR_result WHERE no = " + str(no)
    cur.execute(select_path)
    path = cur.fetchall()[0][0]  

    # 关闭连接
    cur.close()
    conn.close()

    return path


def select_tuning_from_pretrain(pretrain_no):
    # 连接表
    conn = sqlite3.connect('SimCLR_result.db')
    cur = conn.cursor()
    cur.execute('PRAGMA busy_timeout = 10000')

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
    cur.execute('PRAGMA busy_timeout = 10000')

    update_GE = "UPDATE SimCLR_result SET GE = " + str(GE) + ", GE_epoch = " + str(GE_epoch) + " where no = " + str(no)
    cur.execute(update_GE)
    conn.commit()

    # 关闭连接
    cur.close()
    conn.close()


def select_optimism_GE(no):
    # 连接表
    conn = sqlite3.connect('SimCLR_result.db')
    cur = conn.cursor()
    cur.execute('PRAGMA busy_timeout = 10000')

    select_GE = "SELECT GE, GE_epoch FROM SimCLR_result where no = " + str(no)
    cur.execute(select_GE)
    result = cur.fetchall()[0]

    # 关闭连接
    cur.close()
    conn.close()

    return result


def find_no_for_GE(min_no):
    result = None
    try:
        # 打开数据库连接
        conn = sqlite3.connect('SimCLR_result.db')
        conn.execute('PRAGMA busy_timeout = 10000')  # 设置锁超时时间
        cur = conn.cursor()

        # 开始 EXCLUSIVE 事务
        cur.execute('BEGIN EXCLUSIVE TRANSACTION')

        # 查询满足条件的第一条记录
        cur.execute("""
            SELECT no FROM SimCLR_result 
            WHERE GE_epoch = -1 AND type IN ('tuning', 'network') AND no >= ? AND dataset != 'RSA'
            LIMIT 1
        """, (min_no,))
        result = cur.fetchone()

        if result:
            # 使用参数化查询更新记录，避免 SQL 注入
            cur.execute("""
                UPDATE SimCLR_result 
                SET GE_epoch = 0 
                WHERE no = ?;
            """, (result[0],))
            conn.commit()
            print("事务提交成功！")
        else:
            print("未找到满足条件的记录。")
            conn.commit()  # 提交事务，即使没有更新操作
    except Exception as e:
        print(f"事务失败，已回滚：{e}")
        conn.rollback()  # 回滚事务
    finally:
        # 关闭游标和连接
        if 'cur' in locals():
            cur.close()
        conn.close()

    return result[0] if result else None
