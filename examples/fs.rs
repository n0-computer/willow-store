use anyhow::Result;
use std::{
    borrow::Borrow, io, os::unix::fs::MetadataExt, path::PathBuf, str::FromStr, time::Instant,
};

use clap::Parser;
use walkdir::WalkDir;
use willow_store::mock_willow::{TNode, TPoint, WillowValue};
use willow_store::{
    mock_willow::{Subspace, Timestamp},
    BlobSeq, NodeId, QueryRange, QueryRange3d, RedbBlobStore,
};

#[derive(Debug, clap::Parser)]
struct Args {
    #[command(subcommand)]
    subcommand: Subcommand,
}

#[derive(Debug, Clone, clap::Parser)]
enum Subcommand {
    Import(ImportArgs),
    Split(SplitArgs),
}

#[derive(Debug, Clone, clap::Parser)]
struct ImportArgs {
    root: PathBuf,
    #[clap(long)]
    target: PathBuf,
}

#[derive(Debug, Clone, clap::Parser)]
struct SplitArgs {
    #[clap(long, default_value = "3")]
    n: u64,
    #[clap(long)]
    path: PathBuf,
    #[clap(long)]
    root: u64,
}

fn entry_to_triple(entry: walkdir::DirEntry) -> io::Result<Option<(u32, Timestamp, PathBuf)>> {
    let path = entry.path().to_path_buf();
    let metadata = entry.metadata()?;
    if metadata.is_file() {
        // Get creation time using `filetime` crate
        let creation_time: Timestamp = metadata.created().unwrap().into();

        // Get user ID (Unix) or set to 0 (Windows)
        #[cfg(unix)]
        let user_id = metadata.uid();
        #[cfg(not(unix))]
        let user_id = 0u64;

        Ok(Some((user_id, creation_time, path)))
    } else {
        Ok(None) // Skip directories
    }
}

/// Utility to traverse a directory and yield (user_id, creation_time, path) triples
fn traverse(
    root: impl AsRef<std::path::Path>,
) -> impl Iterator<Item = io::Result<(u32, Timestamp, PathBuf)>> {
    let root = root.as_ref().to_path_buf();
    WalkDir::new(root)
        .into_iter()
        .map(|x| x.map_err(io::Error::from))
        .filter_map(move |entry| match entry {
            Ok(entry) => entry_to_triple(entry).transpose(),
            Err(e) => Some(Err(e)),
        })
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    match args.subcommand {
        Subcommand::Import(args) => import(args),
        Subcommand::Split(args) => split(args),
    }
}

fn split(args: SplitArgs) -> Result<()> {
    let node = TNode::from(NodeId::from(args.root));
    let db = RedbBlobStore::new(args.path)?;
    let ss = db.snapshot()?;
    // for item in node.iter(&ss) {
    //     println!("{:?}", item?);
    // }
    for split in node.split_range(QueryRange3d::all(), args.n, &ss) {
        let split = split?;
        println!("{} {}", split.0.pretty(), split.1);
    }
    Ok(())
}

fn import(args: ImportArgs) -> Result<()> {
    let target = args.target;
    std::fs::remove_file(&target).ok();
    let db = RedbBlobStore::new(target)?;
    let mut batch = db.txn()?;
    // let db = MemStore::new();
    // let mut batch = db;
    let mut node = TNode::EMPTY;
    let root = args.root;
    for item in traverse(&root) {
        let (user_id, creation_time, path) = item?;
        let user_id = Subspace::from(user_id as u64);
        let path_rel = path.strip_prefix(&root).unwrap();
        let components = path_rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .map(|x| x.to_string())
            .collect::<Vec<_>>();
        let comp_ref = components.iter().map(|x| x.as_bytes()).collect::<Vec<_>>();
        let wpath = BlobSeq::from(comp_ref.as_slice());
        println!("{} {} {}", user_id, creation_time, wpath);
        let key = TPoint::new(&user_id, &creation_time, wpath.borrow());
        let input = path.to_string_lossy().as_bytes().to_vec();
        // let input = std::fs::read(&path)?;
        node.insert(&key, &WillowValue::hash(&input), &mut batch)?;
    }
    // let db = batch;
    batch.commit()?;
    println!("root id: {}", node.id());
    println!("total count: {}", db.blob_count()?);
    // let ss = db;
    let ss = db.snapshot()?;
    // for item in node.iter(&ss) {
    //     println!("{:?}", item?);
    // }
    let q = QueryRange3d {
        x: QueryRange::all(),
        y: QueryRange::all(),
        z: QueryRange::from(BlobSeq::from_str(r#""arch""#)?..BlobSeq::from_str(r#""arch ""#)?),
    };
    println!("{}", q);
    let t0 = Instant::now();
    let items = node.query(&q, &ss).collect::<Vec<_>>();
    let dt = t0.elapsed();
    let c = items.len();
    // for item in items {
    //     println!("{:?}", item?);
    // }
    // node.dump(&ss)?;
    println!("Elapsed: {} {}", c, dt.as_secs_f64());
    // node.dump(&ss)?;
    for split in node.split_range(QueryRange3d::all(), 3, &ss) {
        println!("{:?}", split?);
    }
    // let (sum, count) = node.average_node_depth(&ss)?;
    // println!("Node count: {}", count);
    // println!("Average Node Depth: {}", (sum as f64) / (count as f64));
    // println!("nodes={} total size={}", ss.size(), ss.total_bytes());
    // let mut total = Duration::ZERO;
    // let n = 100000;
    // for i in 0..n {
    //     let t0 = Instant::now();
    //     let n = node.range_count(&q, &ss)?;
    //     let dt = t0.elapsed();
    //     total += dt;
    //     println!("{} {}", i, dt.as_secs_f64());
    // }
    // println!("Average range count time: {}", (total.as_secs_f64()) / (n as f64));
    Ok(())
}
