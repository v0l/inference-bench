use anyhow::{Result, ensure};

pub const BLOCK_ROWS: usize = 32;
pub const BLOCK_COLUMNS: usize = 2;
pub const TILE_BLOCK_ROWS: usize = 4;
pub const TILE_BLOCK_COLUMNS: usize = 2;
pub const PAGE_TILES: usize = 32;
const OFFSET_BITS: usize = 4;

pub struct CsfPlane<'a> {
    pub rows: usize,
    pub columns: usize,
    pub bases: &'a [u8],
    pub codes: &'a [u8],
    pub page_bases: &'a [u32],
    pub offsets: &'a [u16],
    pub exceptions: &'a [u16],
}

struct Geometry {
    block_columns: usize,
    order: Vec<usize>,
    tile_of: Vec<usize>,
    tiles: usize,
}

fn geometry(rows: usize, columns: usize) -> Geometry {
    let block_rows = rows / BLOCK_ROWS;
    let block_columns = columns.div_ceil(BLOCK_COLUMNS);
    let mut order = Vec::with_capacity(block_rows * block_columns);
    let mut tile_of = Vec::with_capacity(block_rows * block_columns);
    let mut tiles = 0;
    for r in (0..block_rows).step_by(TILE_BLOCK_ROWS) {
        for c in (0..block_columns).step_by(TILE_BLOCK_COLUMNS) {
            for gc in c..(c + TILE_BLOCK_COLUMNS).min(block_columns) {
                for gr in r..(r + TILE_BLOCK_ROWS).min(block_rows) {
                    order.push(gr * block_columns + gc);
                    tile_of.push(tiles);
                }
            }
            tiles += 1;
        }
    }
    Geometry { block_columns, order, tile_of, tiles }
}

impl CsfPlane<'_> {
    pub fn decode(&self) -> Result<Vec<u8>> {
        let (rows, columns) = (self.rows, self.columns);
        ensure!(rows > 0 && rows.is_multiple_of(BLOCK_ROWS) && columns > 0, "csf plane {rows} x {columns} is not whole 32 row blocks");
        ensure!(self.bases.len() == rows, "csf bases have {} entries, expected {rows}", self.bases.len());
        ensure!(self.codes.len() == rows * columns * OFFSET_BITS / 8, "csf codes have {} bytes, expected {}", self.codes.len(), rows * columns * OFFSET_BITS / 8);
        let g = geometry(rows, columns);
        let pages = g.tiles.div_ceil(PAGE_TILES);
        ensure!(self.page_bases.len() == pages, "csf page_bases have {} entries, expected {pages}", self.page_bases.len());
        ensure!(self.offsets.len() == g.order.len() + pages, "csf offsets have {} entries, expected {}", self.offsets.len(), g.order.len() + pages);
        let mut out = vec![0u8; rows * columns];
        let mut field = 0usize;
        for &block in &g.order {
            let (gr, gc) = (block / g.block_columns, block % g.block_columns);
            let width = BLOCK_COLUMNS.min(columns - gc * BLOCK_COLUMNS);
            for row in gr * BLOCK_ROWS..(gr + 1) * BLOCK_ROWS {
                for col in gc * BLOCK_COLUMNS..gc * BLOCK_COLUMNS + width {
                    let nibble = (self.codes[field / 2] >> (4 * (field % 2))) & 0xf;
                    field += 1;
                    let v = self.bases[row] as u16 + nibble as u16;
                    ensure!(v <= 0xff, "csf row {row} base {} plus offset {nibble} passes a byte", self.bases[row]);
                    out[row * columns + col] = v as u8;
                }
            }
        }
        let mut record = 0usize;
        let mut prefix_at_page = vec![0usize; pages];
        for (i, &block) in g.order.iter().enumerate() {
            let page = g.tile_of[i] / PAGE_TILES;
            if i == 0 || g.tile_of[i - 1] / PAGE_TILES != page {
                ensure!(self.offsets[i + page] == 0, "csf page {page} does not start its offsets at zero");
                prefix_at_page[page] = record;
            }
            let at = i + page;
            let (from, to) = (self.offsets[at] as usize, self.offsets[at + 1] as usize);
            ensure!(to >= from, "csf block {block} has a negative exception count");
            let (gr, gc) = (block / g.block_columns, block % g.block_columns);
            let width = BLOCK_COLUMNS.min(columns - gc * BLOCK_COLUMNS);
            for _ in from..to {
                let rec = *self.exceptions.get(record).ok_or_else(|| anyhow::anyhow!("csf exceptions end at {record} records"))?;
                record += 1;
                let local = (rec & 0xff) as usize;
                ensure!(local < BLOCK_ROWS * BLOCK_COLUMNS && local % BLOCK_COLUMNS < width, "csf exception position {local} is outside block {block}");
                let row = gr * BLOCK_ROWS + local / BLOCK_COLUMNS;
                let col = gc * BLOCK_COLUMNS + local % BLOCK_COLUMNS;
                out[row * columns + col] = (rec >> 8) as u8;
            }
        }
        ensure!(record == self.exceptions.len(), "csf has {} exception records, the offsets use {record}", self.exceptions.len());
        for (page, &want) in prefix_at_page.iter().enumerate() {
            ensure!(self.page_bases[page] as usize == 2 * want, "csf page {page} base {} is not byte offset {}", self.page_bases[page], 2 * want);
        }
        Ok(out)
    }
}
