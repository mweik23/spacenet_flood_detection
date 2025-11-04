bad="mitchweikert@macbookpro.mynetworksettings.com"
good="mweikert2394@gmail.com"

git filter-branch --env-filter '
if [ "$GIT_AUTHOR_EMAIL" = "'"$bad"'" ]; then
  export GIT_AUTHOR_EMAIL="'"$good"'"
  export GIT_COMMITTER_EMAIL="'"$good"'"
fi
' -- --all

git push --force-with-lease --all
git push --force-with-lease --tags