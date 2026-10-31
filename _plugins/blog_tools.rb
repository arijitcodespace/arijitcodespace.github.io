# frozen_string_literal: true
#
# Blog tools for paper write-ups. Runs at build time (GitHub Actions build).
# Active on every post, and on any other page whose front matter has `math: true`
# (a post can opt out with `math: false`).
#
# 1. Math, written the LaTeX way. Before Markdown runs, these are turned into the
#    form kramdown + MathJax expect, so subscripts and `\\` survive intact:
#      $x$   \(x\)            -> inline math
#      $$x$$ (on its own line) \[x\]  \begin{equation|align|gather|...}  -> display math
#    `$5 and $10` stays as text (a closing $ followed by a digit is not math),
#    and nothing inside code blocks or `inline code` is touched.
#
# 2. Citations, from _bibliography/references.bib (or `bibliography: name.bib`
#    in a post's front matter):
#      \cite{key} \cite{a,b} \citep{key} \cite[p.~5]{key}  -> [1]  [1, 2]  [1, p. 5]
#      \citet{key}                                         -> Vaswani et al. [1]
#    Numbers follow the order of first citation, every citation links to its entry,
#    and a "References" list is added at the end of the post automatically.
#
# 3. Figures from {% include figure.html ... label="fig:x" %} are numbered in order,
#    and \ref{fig:x} becomes a link to the figure. \eqref{eq:x} / \ref{eq:x} are
#    left for MathJax, which numbers and links equations itself.

require "cgi"

module BlogTools
  # ---------------------------------------------------------------- helpers ---

  def self.enabled?(doc)
    flag = doc.data["math"]
    post = doc.respond_to?(:collection) && doc.collection && doc.collection.label == "posts"
    post ? flag != false : flag == true
  end

  def self.markdown?(doc)
    ext = doc.respond_to?(:extname) ? doc.extname : doc.ext
    %w[.md .markdown .mkd .mkdn .mdown].include?(ext.to_s.downcase)
  end

  def self.esc(text)
    CGI.escapeHTML(text.to_s)
  end

  # ------------------------------------------------- 1. math normalization ---

  BLANK_LINE = "(?:(?!\\n[ \\t]*\\n).)"
  MATH_ENVS = "equation|align|gather|multline|flalign|alignat|eqnarray"

  SOURCE_TOKENS = Regexp.new([
    # fenced code blocks
    "(?<fence>^[ \\t]{0,3}(?<fchar>`{3,}|~{3,})[^\\n]*\\n.*?(?:^[ \\t]{0,3}\\k<fchar>[`~]*[ \\t]*$|\\z))",
    # Liquid blocks whose contents must stay verbatim
    "(?<liquid>\\{%-?\\s*(?<lq>raw|comment|highlight)\\b.*?\\{%-?\\s*end\\k<lq>\\s*-?%\\})",
    "(?<html_comment><!--.*?-->)",
    # inline code spans
    "(?<code>(?<!`)(?<ticks>`+)(?!`)#{BLANK_LINE}+?(?<!`)\\k<ticks>(?!`))",
    # math in all the LaTeX spellings
    "(?<dmath>(?<!\\\\)\\$\\$#{BLANK_LINE}+?\\$\\$)",
    "(?<env>\\\\begin\\{(?<envname>#{MATH_ENVS})(?<star>\\*?)\\}.*?\\\\end\\{\\k<envname>\\k<star>\\})",
    "(?<bracket>(?<!\\\\)\\\\\\[#{BLANK_LINE}+?\\\\\\])",
    "(?<paren>(?<!\\\\)\\\\\\(#{BLANK_LINE}+?\\\\\\))",
    "(?<smath>(?<![\\\\$])\\$(?![\\s$])(?:[^$\\\\\\n`]|\\\\.|\\n(?![ \\t]*\\n))*?(?<![\\s\\\\])\\$(?![\\d$]))",
    # reference commands outside math: protect _ and * in their keys from Markdown
    "(?<refcmd>\\\\(?:cite[tp]?|ref|eqref|autoref)(?:\\[[^\\]\\n]*\\])?\\{[^}\\n]*\\})",
  ].join("|"), Regexp::MULTILINE)

  # Keep math from being read as Liquid: `{{` / `{%` -> `{ {` / `{ %` (same TeX).
  def self.liquid_safe(math)
    math.gsub("{{", "{ {").gsub("{%", "{ %")
  end

  def self.display(body)
    "\n\n$$#{liquid_safe(body)}$$\n\n"
  end

  def self.normalize_math(source)
    source.gsub(SOURCE_TOKENS) do
      m = Regexp.last_match
      if m[:dmath]
        body = m[:dmath][2..-3]
        line_before = m.pre_match[/[^\n]*\z/]
        line_after = m.post_match[/\A[^\n]*/]
        if line_before.strip.empty? && line_after.strip.empty?
          display(body)
        else
          "$$#{liquid_safe(body)}$$"
        end
      elsif m[:env]
        display("\n#{m[:env]}\n")
      elsif m[:bracket]
        display(m[:bracket][2..-3])
      elsif m[:paren]
        "$$#{liquid_safe(m[:paren][2..-3])}$$"
      elsif m[:smath]
        "$$#{liquid_safe(m[:smath][1..-2])}$$"
      elsif m[:refcmd]
        m[:refcmd].gsub(/\{[^}]*\}\z/) { |arg| arg.gsub(/(?<!\\)([_*])/) { "\\#{Regexp.last_match(1)}" } }
      else
        m[0] # code, Liquid blocks, comments: untouched
      end
    end
  end

  # ------------------------------------------------------------- 2. BibTeX ---

  MONTHS = %w[jan feb mar apr may jun jul aug sep oct nov dec]
             .zip(%w[January February March April May June July August September October November December]).to_h

  ACCENTS = {
    "'" => "\u0301", "`" => "\u0300", "^" => "\u0302", '"' => "\u0308", "~" => "\u0303",
    "=" => "\u0304", "." => "\u0307", "u" => "\u0306", "v" => "\u030C", "H" => "\u030B",
    "c" => "\u0327", "k" => "\u0328", "r" => "\u030A",
  }.freeze

  SPECIAL = {
    "ss" => "ß", "ae" => "æ", "AE" => "Æ", "oe" => "œ", "OE" => "Œ", "aa" => "å", "AA" => "Å",
    "l" => "ł", "L" => "Ł", "o" => "ø", "O" => "Ø", "i" => "i", "j" => "j",
  }.freeze

  @bib_cache = {}

  def self.bibliography(site, name)
    path = File.join(site.source, "_bibliography", name)
    unless File.exist?(path)
      Jekyll.logger.warn "Blog tools:", "bibliography file _bibliography/#{name} not found"
      return {}
    end
    mtime = File.mtime(path)
    cached = @bib_cache[path]
    return cached[:entries] if cached && cached[:mtime] == mtime

    entries = parse_bibtex(File.read(path, encoding: "UTF-8"))
    @bib_cache[path] = { mtime: mtime, entries: entries }
    entries
  end

  def self.parse_bibtex(text)
    entries = {}
    macros = MONTHS.dup
    pos = 0
    while (at = text.index("@", pos))
      head = text.match(/\G@\s*([A-Za-z]+)\s*\{/, at)
      unless head
        pos = at + 1
        next
      end
      type = head[1].downcase
      start = head.end(0)
      depth = 1
      i = start
      while i < text.length && depth.positive?
        case text[i]
        when "\\" then i += 1
        when "{" then depth += 1
        when "}" then depth -= 1
        end
        i += 1
      end
      body = text[start...(i - 1)]
      pos = i
      next if %w[comment preamble].include?(type)

      if type == "string"
        parse_fields(body, macros).each { |k, v| macros[k] = v }
        next
      end
      key, rest = body.split(",", 2)
      next unless key && rest

      entries[key.strip] = { "type" => type, "fields" => parse_fields(rest, macros) }
    end
    entries
  end

  def self.parse_fields(str, macros)
    fields = {}
    n = str.length
    i = 0
    loop do
      i += 1 while i < n && str[i] =~ /[\s,]/
      break if i >= n

      m = str.match(/\G([A-Za-z0-9_\-:.+]+)\s*=\s*/, i)
      break unless m

      name = m[1].downcase
      i = m.end(0)
      value = +""
      loop do
        if str[i] == "{"
          depth = 1
          j = i + 1
          while j < n && depth.positive?
            if str[j] == "\\"
              j += 2
              next
            end
            depth += 1 if str[j] == "{"
            depth -= 1 if str[j] == "}"
            j += 1
          end
          value << str[(i + 1)...(j - 1)].to_s
          i = j
        elsif str[i] == '"'
          depth = 0
          j = i + 1
          while j < n && !(str[j] == '"' && depth.zero?)
            if str[j] == "\\"
              j += 2
              next
            end
            depth += 1 if str[j] == "{"
            depth -= 1 if str[j] == "}"
            j += 1
          end
          value << str[(i + 1)...j].to_s
          i = j + 1
        else
          word = str.match(/\G([^\s,#}]+)/, i)
          break unless word

          value << (macros[word[1].downcase] || word[1])
          i = word.end(0)
        end
        i += 1 while i < n && str[i] =~ /\s/
        break unless str[i] == "#"

        i += 1
        i += 1 while i < n && str[i] =~ /\s/
      end
      fields[name] = value
    end
    fields
  end

  # LaTeX-ish BibTeX text -> plain Unicode (math kept as \( ... \) for MathJax).
  def self.decode(raw)
    s = raw.to_s.dup
    maths = []
    s = s.gsub(/(?<!\\)\$(.+?)(?<!\\)\$/m) do
      maths << Regexp.last_match(1)
      "#{maths.size - 1}" # private-use characters: survive strip/squeeze
    end
    s = s.gsub("\\{", "\u0001").gsub("\\}", "\u0002")
    s = s.gsub(/\\url\s*\{([^}]*)\}/) { Regexp.last_match(1) }
    8.times do
      break unless s.gsub!(/\\(?:textit|textbf|textsc|textrm|textsf|texttt|textnormal|textup|emph|mbox|text|mathrm|mathbf|mathit|mathsf|mathtt)\s*\{([^{}]*)\}/) { Regexp.last_match(1) }
    end
    s = s.gsub(/\\([cvuHkr])\s*\{\s*\\?([A-Za-z])\s*\}/) { accent(Regexp.last_match(1), Regexp.last_match(2)) }
    s = s.gsub(/\\([cvuHkr])\s+([A-Za-z])/) { accent(Regexp.last_match(1), Regexp.last_match(2)) }
    s = s.gsub(/\\(['`^"~=.])\s*\{\s*\\?([A-Za-z])\s*\}/) { accent(Regexp.last_match(1), Regexp.last_match(2)) }
    s = s.gsub(/\\(['`^"~=.])\s*\\?([A-Za-z])/) { accent(Regexp.last_match(1), Regexp.last_match(2)) }
    s = s.gsub(/\\(ss|ae|AE|oe|OE|aa|AA|l|L|o|O|i|j)(?![A-Za-z]) ?/) { SPECIAL[Regexp.last_match(1)] }
    s = s.gsub(/\\([&%$#_])/) { Regexp.last_match(1) }
    s = s.gsub("---", "\u2014").gsub("--", "\u2013")
    s = s.gsub("~", " ").gsub(/\\[,;: ]/, " ")
    s = s.gsub(/\\([A-Za-z]+)/) { Regexp.last_match(1) }
    s = s.delete("{}").tr("\u0001\u0002", "{}")
    s = s.gsub(/\s+/, " ").strip
    s = s.gsub(/(\d+)/) { "\\(#{maths[Regexp.last_match(1).to_i]}\\)" }
    s.unicode_normalize(:nfc)
  end

  def self.accent(cmd, letter)
    letter + ACCENTS.fetch(cmd, "")
  end

  # Split on a separator only at brace depth 0.
  def self.split_top(str, separator)
    parts = []
    depth = 0
    cur = +""
    i = 0
    while i < str.length
      c = str[i]
      depth += 1 if c == "{"
      depth -= 1 if c == "}"
      if depth.zero? && (sep = str[i..].match(separator)) && sep.begin(0).zero?
        parts << cur
        cur = +""
        i += sep[0].length
        next
      end
      cur << c
      i += 1
    end
    parts << cur
    parts.map(&:strip).reject(&:empty?)
  end

  def self.parse_name(raw)
    return { "others" => true } if raw.strip.casecmp?("others")

    commas = split_top(raw, /,/)
    if commas.length >= 2
      last = decode(commas[0])
      first = decode(commas[-1])
      jr = commas.length >= 3 ? decode(commas[1]) : nil
      full = [first, last].reject(&:empty?).join(" ")
      full += ", #{jr}" if jr && !jr.empty?
      return { "last" => last, "full" => full }
    end

    words = split_top(raw, /\s+/)
    if words.length <= 1
      name = decode(raw)
      return { "last" => name, "full" => name }
    end
    von = (1...(words.length - 1)).find { |k| words[k] =~ /\A[a-z]/ } || (words.length - 1)
    first = decode(words[0...von].join(" "))
    last = decode(words[von..].join(" "))
    { "last" => last, "full" => [first, last].reject(&:empty?).join(" ") }
  end

  def self.authors(fields)
    split_top(fields["author"] || fields["editor"] || "", /\s+and\s+/i).map { |n| parse_name(n) }
  end

  def self.with_period(text)
    text =~ /[.?!]\z/ ? text : "#{text}."
  end

  def self.arxiv_id(fields)
    if fields["eprint"] && (fields["archiveprefix"].to_s =~ /arxiv/i || fields["eprinttype"].to_s =~ /arxiv/i)
      return fields["eprint"].strip
    end

    [fields["journal"], fields["note"], fields["howpublished"]].compact.each do |f|
      id = f[/arXiv:\s*([0-9]{4}\.[0-9]{4,5})/i, 1]
      return id if id
    end
    nil
  end

  def self.link(fields)
    candidates = []
    candidates << decode(fields["url"]) if fields["url"]
    if fields["doi"]
      doi = decode(fields["doi"])
      candidates << (doi =~ %r{\Ahttps?://} ? doi : "https://doi.org/#{doi}")
    end
    id = arxiv_id(fields)
    candidates << "https://arxiv.org/abs/#{id}" if id
    candidates.find { |u| u =~ %r{\Ahttps?://\S+\z} }
  end

  def self.venue(entry)
    f = entry["fields"]
    v = case entry["type"]
        when "article" then f["journal"]
        when "inproceedings", "conference", "incollection" then f["booktitle"]
        when "phdthesis" then ["PhD thesis", f["school"]].compact.join(", ")
        when "mastersthesis" then ["Master's thesis", f["school"]].compact.join(", ")
        when "techreport" then f["institution"]
        when "book" then f["publisher"]
        end
    v ||= f["journal"] || f["booktitle"] || f["howpublished"] || f["publisher"] || f["institution"]
    v = decode(v) if v
    v = nil if v && (v.empty? || v =~ %r{\Ahttps?://})
    id = arxiv_id(f)
    v ||= "arXiv:#{id}" if id
    v
  end

  def self.year(fields)
    (fields["year"] || fields["date"].to_s[0, 4]).to_s.strip
  end

  def self.author_list(names)
    real = names.reject { |a| a["others"] }
    truncated = names.any? { |a| a["others"] } || real.length > 6
    shown = (truncated && real.length > 3 ? real.first(3) : real).map { |a| a["full"] }
    return "" if shown.empty?
    return "#{shown.join(', ')}#{',' if shown.length > 1} et al." if truncated

    case shown.length
    when 1 then shown[0]
    when 2 then "#{shown[0]} and #{shown[1]}"
    else "#{shown[0..-2].join(', ')}, and #{shown[-1]}"
    end
  end

  def self.short_authors(names)
    real = names.reject { |a| a["others"] }
    return "Anonymous" if real.empty?
    return "#{real[0]['last']} et al." if real.length > 2 || names.any? { |a| a["others"] }
    return real[0]["last"] if real.length == 1

    "#{real[0]['last']} and #{real[1]['last']}"
  end

  def self.reference_html(key, entry, backlink)
    f = entry["fields"]
    names = authors(f)
    title = decode(f["title"] || key)
    url = link(f)
    venue = venue(entry)
    yr = year(f)

    parts = []
    list = author_list(names)
    parts << %(<span class="ref-authors">#{esc(with_period(list))}</span>) unless list.empty?
    title_html = esc(with_period(title))
    parts << if url
               %(<a class="ref-title" href="#{esc(url)}">#{title_html}</a>)
             else
               %(<span class="ref-title">#{title_html}</span>)
             end
    tail = [venue && "<em>#{esc(venue)}</em>", (esc(yr) unless yr.empty?)].compact.join(", ")
    parts << %(<span class="ref-venue">#{with_period(tail)}</span>) unless tail.empty?
    parts << %(<a class="ref-back" href="##{esc(backlink)}" title="Back to the text" aria-label="Back to the text">&#8617;</a>)
    %(<li id="ref-#{esc(key)}">#{parts.join(' ')}</li>)
  end

  def self.tooltip(entry)
    f = entry["fields"]
    yr = year(f)
    "#{short_authors(authors(f))}#{" (#{yr})" unless yr.empty?}. #{decode(f['title'].to_s)}"
  end

  # --------------------------------------------- 3. figures, cites, refs ---

  CODE_BLOCKS = %r{<pre\b.*?</pre>|<code\b.*?</code>|<script\b.*?</script>|<style\b.*?</style>}m
  CODE_SKIP = /(#{CODE_BLOCKS})/m
  # Code, any HTML tag (so attributes are never rewritten), and math.
  TEXT_SKIP = /(#{CODE_BLOCKS}|<[^>]*>|\\\(.*?\\\)|\\\[.*?\\\])/m
  FIGURE_NUM = %r{(<figure class="figure"[^>]*>.*?<span class="figure-num">)\s*(</span>)}m
  CITE = /\\(cite[tp]?)(?:\[([^\]]*)\])?\{([^}]*)\}/
  REF = /\\(ref|autoref)\{([^}]*)\}/
  ANY_REF = /\s*\\(?:cite[tp]?|ref|eqref|autoref)(?:\[[^\]]*\])?\{[^}]*\}/

  # Alt text comes from the caption; drop citation/reference commands from it.
  def self.clean_alt(tag)
    tag.gsub(/\balt="([^"]*)"/) do
      raw = Regexp.last_match(1)
      text = raw.gsub(ANY_REF, "").tr("~", " ").gsub(/\s+([.,;:])/, '\1').squeeze(" ").strip
      %(alt="#{text}")
    end
  end

  # Apply a block to every part of the HTML outside code (and, by default, math).
  def self.each_text_segment(html, skip = TEXT_SKIP)
    html.split(skip).each_with_index.map do |part, idx|
      idx.odd? ? part : yield(part)
    end.join
  end

  DISPLAY_MATH = /^\\\[.*?\\\]$/m

  def self.process_html(doc)
    html = doc.content

    # Give each displayed equation its own sideways-scrolling box, so a wide (or
    # numbered) equation never makes the whole page wider than a phone screen.
    if html.include?("\\[")
      html = each_text_segment(html, CODE_SKIP) do |text|
        text.gsub(DISPLAY_MATH) { |math| %(<div class="math-display">#{math}</div>) }
      end
      doc.content = html
    end

    return unless html.include?("\\cite") || html.include?("\\ref") || html.include?("\\autoref") ||
                  html.include?('class="figure"')

    where = doc.relative_path

    # Figures: number them in order and remember their labels.
    figures = {}
    count = 0
    html = each_text_segment(html, CODE_SKIP) do |text|
      text.gsub(FIGURE_NUM) do
        head = Regexp.last_match(1)
        close = Regexp.last_match(2)
        count += 1
        label = head[/\A<figure class="figure"[^>]*\bid="([^"]+)"/, 1]
        figures[CGI.unescapeHTML(label)] = count if label
        "#{clean_alt(head)}#{count}#{close}"
      end
    end

    # LaTeX non-breaking spaces before references: Figure~\ref{...}
    html = each_text_segment(html) { |text| text.gsub(/~(?=\\(?:cite[tp]?|ref|eqref|autoref)\b)/, "&nbsp;") }

    # \ref{fig:...} -> figure number. Other labels are left for MathJax (equations).
    html = each_text_segment(html) do |text|
      text.gsub(REF) do |whole|
        cmd = Regexp.last_match(1)
        label = CGI.unescapeHTML(Regexp.last_match(2).strip)
        if figures.key?(label)
          num = figures[label]
          text_out = cmd == "autoref" ? "Figure&nbsp;#{num}" : num.to_s
          %(<a class="xref" href="##{esc(label)}">#{text_out}</a>)
        elsif label.start_with?("fig:")
          Jekyll.logger.warn "Blog tools:", "#{where}: no figure labelled '#{label}'"
          %(<span class="xref-missing" title="No figure labelled #{esc(label)}">??</span>)
        else
          whole
        end
      end
    end

    # Citations
    bib = nil
    order = []
    html = each_text_segment(html) do |text|
      text.gsub(CITE) do
        cmd = Regexp.last_match(1)
        note = Regexp.last_match(2)
        keys = Regexp.last_match(3).split(",").map { |k| CGI.unescapeHTML(k.strip) }.reject(&:empty?)
        bib ||= bibliography(doc.site, doc.data["bibliography"] || "references.bib")
        pieces = keys.map do |key|
          entry = bib[key]
          unless entry
            Jekyll.logger.warn "Blog tools:", "#{where}: no BibTeX entry for '#{key}'"
            next [key, %(<span class="citation-missing" title="Missing BibTeX entry: #{esc(key)}">?</span>)]
          end
          first_use = !order.include?(key)
          order << key if first_use
          num = order.index(key) + 1
          id_attr = first_use ? %( id="cite-#{esc(key)}") : ""
          [key, %(<a href="#ref-#{esc(key)}"#{id_attr} title="#{esc(tooltip(entry))}">#{num}</a>)]
        end
        note_html = note && !note.strip.empty? ? ", #{esc(note.tr('~', ' ').strip)}" : ""
        if cmd == "citet"
          pieces.map do |key, link|
            name = bib[key] ? esc(short_authors(authors(bib[key]["fields"]))) : esc(key)
            %(#{name} <span class="citation">[#{link}#{note_html}]</span>)
          end.join(", ")
        else
          %(<span class="citation">[#{pieces.map(&:last).join(', ')}#{note_html}]</span>)
        end
      end
    end

    unless order.empty?
      items = order.map { |key| reference_html(key, bib[key], "cite-#{key}") }
      html += %(\n<section class="references" id="references">\n<h2>References</h2>\n<ol class="reference-list">\n#{items.join("\n")}\n</ol>\n</section>\n)
    end

    doc.content = html
  end
end

Jekyll::Hooks.register [:pages, :documents], :pre_render do |doc, _payload|
  next unless BlogTools.enabled?(doc) && BlogTools.markdown?(doc)

  doc.content = BlogTools.normalize_math(doc.content)
end

Jekyll::Hooks.register [:pages, :documents], :post_convert do |doc|
  next unless BlogTools.enabled?(doc)

  BlogTools.process_html(doc)
end
