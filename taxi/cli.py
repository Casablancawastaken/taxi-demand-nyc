import argparse
import json
from .db import connect,reset_data, data_mode
from .sources import generate_demo,sync_real
from .modeling import training_pipeline,produce_forecast,reproduce_experiment,rollback
from .monitoring import run_monitor


def main():
    parser=argparse.ArgumentParser(prog='python -m taxi.cli',description='Прогноз спроса на такси: вариант 16')
    commands=parser.add_subparsers(dest='command',required=True)
    demo=commands.add_parser('demo')
    demo.add_argument('--reset',action='store_true')
    demo.add_argument('--end-date',default='2026-11-02')
    demo.add_argument('--days',type=int,default=240)
    sync=commands.add_parser('sync')
    sync.add_argument('--start',default='2024-01')
    sync.add_argument('--months',type=int,default=6)
    train=commands.add_parser('train')
    train.add_argument('--max-zones',type=int,default=None)
    train.add_argument('--force',action='store_true')
    forecast=commands.add_parser('forecast')
    forecast.add_argument('--days',type=int,default=7)
    monitor=commands.add_parser('monitor')
    monitor.add_argument('--simulate-drift',action='store_true')
    reproduce=commands.add_parser('reproduce')
    reproduce.add_argument('experiment_id')
    reg=commands.add_parser('registry')
    rollback_parser=commands.add_parser('rollback')
    rollback_parser.add_argument('version')
    commands.add_parser('reset')
    args=parser.parse_args()
    conn=connect()
    try:
        if args.command=='reset':
            reset_data(conn)
            result={'reset':True}
        elif args.command=='demo':
            generate_demo(conn,args.end_date,args.days,reset=args.reset)
            result={'mode':'demo','trips':conn.execute('SELECT COUNT(*) FROM trips').fetchone()[0]}
        elif args.command=='sync':
            result=sync_real(conn,args.start,args.months)
        elif args.command=='train':
            result=training_pipeline(conn,args.max_zones,args.force)
        elif args.command=='forecast':
            result=produce_forecast(conn,args.days)
        elif args.command=='monitor':
            result=run_monitor(conn,args.simulate_drift)
        elif args.command=='reproduce':
            result=reproduce_experiment(conn,args.experiment_id)
        elif args.command=='registry':
            result=[dict(x) for x in conn.execute('SELECT version,created_at,selected_model,active,data_signature FROM models ORDER BY created_at DESC')]
        else:
            result={'active_version':rollback(conn,args.version)}
        print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    finally:
        conn.close()

if __name__=='__main__':
    main()
